//! Binary translation

use crate::{engine::Engine, riscv};
use anyhow::{anyhow, bail, Context, Result};
use llvm_sys::{core::*, prelude::*, LLVMIntPredicate::*};
use std::{
    cell::Cell,
    collections::{BTreeSet, HashMap},
    ffi::CString,
};

static NONAME: &'static i8 = unsafe { std::mem::transmute("\0".as_ptr()) };

/// A translator for an entire ELF file.
pub struct ElfTranslator<'a> {
    pub elf: &'a elf::File,
    /// The translation engine.
    pub engine: &'a Engine,
    /// Predicted branch target addresses.
    pub target_addrs: BTreeSet<u64>,
    /// Basic blocks for each instruction address.
    pub inst_bbs: HashMap<u64, LLVMBasicBlockRef>,
}

impl<'a> ElfTranslator<'a> {
    /// Create a new ELF file translator.
    pub fn new(elf: &'a elf::File, engine: &'a Engine) -> Self {
        Self {
            elf,
            engine,
            target_addrs: Default::default(),
            inst_bbs: Default::default(),
        }
    }

    /// Get an iterator over the sections in the binary.
    pub fn sections(&self) -> impl Iterator<Item = &'a elf::Section> + '_ {
        self.elf
            .sections
            .iter()
            .filter(|section| (section.shdr.flags.0 & elf::types::SHF_EXECINSTR.0) != 0)
    }

    /// Get an iterator over the instructions in a section.
    pub fn instructions(
        &self,
        section: &'a elf::Section,
    ) -> impl Iterator<Item = (u64, riscv::Format)> + '_ {
        section
            .data
            .chunks(4)
            .enumerate()
            .map(move |(i, raw)| (section.shdr.addr + i as u64 * 4, riscv::parse(raw)))
    }

    /// Get an iterator over all instructions in the binary.
    pub fn all_instructions(&self) -> impl Iterator<Item = (u64, riscv::Format)> + '_ {
        self.sections().flat_map(move |s| self.instructions(s))
    }

    /// Analyze the binary and estimate the set of possible branch target
    /// addresses.
    pub fn update_target_addrs(&mut self) {
        let mut target_addrs = BTreeSet::new();

        // Ensure that we can jump to the entry symbol.
        target_addrs.insert(self.elf.ehdr.entry);

        // Ensure that we can jump to the beginning of a section.
        for section in self.sections() {
            target_addrs.insert(section.shdr.addr);
        }

        // Estimate target addresses.
        for (addr, inst) in self.all_instructions() {
            match inst {
                riscv::Format::Imm12RdRs1(
                    fmt
                    @
                    riscv::FormatImm12RdRs1 {
                        op: riscv::OpcodeImm12RdRs1::Jalr,
                        ..
                    },
                ) => {
                    debug!("Found register jump 0x{:x}: {}", addr, inst);

                    // If we keep the PC around, we expect to jump back to the
                    // next instruction at some point.
                    if fmt.rd != 0 {
                        target_addrs.insert(addr + 4);
                    }
                }
                riscv::Format::Jimm20Rd(
                    fmt
                    @
                    riscv::FormatJimm20Rd {
                        op: riscv::OpcodeJimm20Rd::Jal,
                        ..
                    },
                ) => {
                    // Ensure that we can branch to the target address.
                    let target = (addr as i64).wrapping_add(fmt.jimm() as i64) as u64;
                    debug!(
                        "Found immediate jump 0x{:x}: {} to 0x{:x}",
                        addr, inst, target,
                    );
                    target_addrs.insert(target);

                    // If we keep the PC around, we expect to jump back to the
                    // next instruction at some point.
                    if fmt.rd != 0 {
                        target_addrs.insert(addr + 4);
                    }
                }
                riscv::Format::Bimm12hiBimm12loRs1Rs2(fmt) => {
                    let target = (addr as i64).wrapping_add(fmt.bimm() as i64) as u64;
                    debug!("Found branch 0x{:x}: {} to 0x{:x}", addr, inst, target,);
                    target_addrs.insert(target);
                    target_addrs.insert(addr + 4);
                }
                _ => (),
            }
        }

        // Dump what we have found.
        debug!("Predicted jump targets:");
        for &addr in &target_addrs {
            debug!("  - 0x{:x}", addr);
        }

        self.target_addrs = target_addrs;
    }

    /// Translate the binary.
    pub fn translate(&mut self) -> Result<()> {
        unsafe { self.translate_inner() }
    }

    unsafe fn translate_inner(&mut self) -> Result<()> {
        debug!("Translating binary");
        let builder = LLVMCreateBuilderInContext(self.engine.context);

        // Assemble the struct type which holds the CPU state.
        let state_type =
            LLVMStructCreateNamed(self.engine.context, format!("cpu\0").as_ptr() as *const _);
        let mut state_fields = [
            LLVMPointerType(LLVMInt8Type(), 0), // Context
            LLVMArrayType(LLVMInt32Type(), 32), // Registers
            LLVMInt32Type(),                    // PC
            LLVMInt64Type(),                    // Retired Instructions
        ];
        LLVMStructSetBody(
            state_type,
            state_fields.as_mut_ptr(),
            state_fields.len() as u32,
            0,
        );
        let state_ptr_type = LLVMPointerType(state_type, 0u32);

        // Declare the callbacks.
        self.declare_func(
            "banshee_load",
            LLVMInt32Type(),
            [
                state_ptr_type,  // CPU
                LLVMInt32Type(), // Address
                LLVMInt8Type(),  // Size
            ],
        );
        self.declare_func(
            "banshee_store",
            LLVMVoidType(),
            [
                state_ptr_type,  // CPU
                LLVMInt32Type(), // Address
                LLVMInt32Type(), // Value
                LLVMInt8Type(),  // Size
            ],
        );
        self.declare_func(
            "banshee_csr_read",
            LLVMInt32Type(),
            [
                state_ptr_type,  // CPU
                LLVMInt16Type(), // CSR
            ],
        );
        self.declare_func(
            "banshee_csr_write",
            LLVMVoidType(),
            [
                state_ptr_type,  // CPU
                LLVMInt16Type(), // CSR
                LLVMInt32Type(), // Value
            ],
        );

        // Emit the function which will run the binary.
        let func_name = format!("execute_binary\0");
        let func_type = LLVMFunctionType(LLVMVoidType(), [state_ptr_type].as_mut_ptr(), 1, 0);
        let func = LLVMAddFunction(
            self.engine.module,
            func_name.as_ptr() as *const _,
            func_type,
        );
        let state_ptr = LLVMGetParam(func, 0);

        // Create the entry block.
        let entry_bb = LLVMAppendBasicBlockInContext(
            self.engine.context,
            func,
            b"entry\0".as_ptr() as *const _,
        );

        // Gather the set of executable addresses.
        let inst_addrs: BTreeSet<u64> = self
            .all_instructions()
            .map(|(addr, _)| addr)
            .chain(self.sections().map(|section| section.shdr.addr))
            .chain(
                self.sections()
                    .map(|section| section.shdr.addr + section.shdr.size),
            )
            .collect();

        // Create a basic block for every instruction address.
        let inst_bbs: HashMap<u64, LLVMBasicBlockRef> = inst_addrs
            .iter()
            .map(|&addr| {
                let bb = LLVMAppendBasicBlockInContext(
                    self.engine.context,
                    func,
                    format!("inst_0x{:x}\0", addr).as_ptr() as *const _,
                );
                (addr, bb)
            })
            .collect();
        self.inst_bbs = inst_bbs;

        // Emit the branch to the entry symbol.
        LLVMPositionBuilderAtEnd(builder, entry_bb);
        LLVMBuildBr(builder, self.inst_bbs[&self.elf.ehdr.entry]);

        // Create a translator for each section.
        let section_tran: Vec<_> = self
            .sections()
            .map(|section| SectionTranslator {
                elf: self,
                section,
                engine: self.engine,
                func,
                state_ptr,
                builder,
                addr_start: section.shdr.addr,
                addr_end: section.shdr.addr + section.shdr.size,
            })
            .collect();

        // Emit the instructions for each section.
        for tran in &section_tran {
            tran.emit()?;
        }

        // Emit escape abort code into all unpopulated instruction locations.
        for (&addr, &bb) in &self.inst_bbs {
            if LLVMGetBasicBlockTerminator(bb).is_null() {
                trace!("Plugging instruction slot hole at 0x{:x}", addr);
                LLVMPositionBuilderAtEnd(builder, bb);
                section_tran[0].emit_escape_abort(addr);
            }
        }

        // Clean up.
        LLVMDisposeBuilder(builder);
        Ok(())
    }

    unsafe fn declare_func(
        &self,
        name: &str,
        ret: LLVMTypeRef,
        args: impl AsRef<[LLVMTypeRef]>,
    ) -> LLVMValueRef {
        let args = args.as_ref();
        LLVMAddFunction(
            self.engine.module,
            CString::new(name).unwrap().as_ptr() as *const _,
            LLVMFunctionType(ret, args.as_ptr() as *mut _, args.len() as u32, 0),
        )
    }
}

pub struct SectionTranslator<'a> {
    elf: &'a ElfTranslator<'a>,
    section: &'a elf::Section,
    engine: &'a Engine,
    func: LLVMValueRef,
    /// An LLVM value that holds the pointer to the CPU state structure.
    state_ptr: LLVMValueRef,
    /// The builder to emit instructions with.
    builder: LLVMBuilderRef,
    /// The first address in the section.
    addr_start: u64,
    /// The point beyond the last address in the section.
    addr_end: u64,
}

impl<'a> SectionTranslator<'a> {
    /// Emit the code to handle the case when the PC lands outside the sections
    /// of the binary.
    unsafe fn emit_escape_abort(&self, addr: u64) {
        trace!("Emit escape abort at 0x{:x}", addr);
        LLVMBuildRetVoid(self.builder);
    }

    /// Emit the code to handle an illegal instruction.
    unsafe fn emit_illegal_abort(&self, addr: u64) {
        trace!("Emit illegal instruction abort at 0x{:x}", addr);
        LLVMBuildRetVoid(self.builder);
    }

    /// Emit the code to handle a branch to an unpredicted instruction.
    unsafe fn emit_branch_abort(&self, inst_addr: u64, _target: LLVMValueRef) {
        trace!("Emit illegal branch abort at 0x{:x}", inst_addr);
        LLVMBuildRetVoid(self.builder);
    }

    /// Emit the code for the entire section.
    unsafe fn emit(&self) -> Result<()> {
        for (addr, inst) in self.elf.instructions(self.section) {
            let tran = InstructionTranslator {
                section: self,
                builder: self.builder,
                addr,
                inst,
                was_terminator: Default::default(),
            };
            LLVMPositionBuilderAtEnd(self.builder, self.elf.inst_bbs[&addr]);
            match tran.emit() {
                Ok(()) => (),
                Err(e) => {
                    error!("{}", e);
                    self.emit_illegal_abort(addr);
                }
            }
            if LLVMGetBasicBlockTerminator(LLVMGetInsertBlock(self.builder)).is_null() {
                LLVMBuildBr(self.builder, self.elf.inst_bbs[&(addr + 4)]);
            }
        }
        Ok(())
    }
}

pub struct InstructionTranslator<'a> {
    section: &'a SectionTranslator<'a>,
    builder: LLVMBuilderRef,
    addr: u64,
    inst: riscv::Format,
    was_terminator: Cell<bool>,
}

impl<'a> InstructionTranslator<'a> {
    unsafe fn emit(&self) -> Result<()> {
        // Update the PC register to reflect this instruction.
        LLVMBuildStore(
            self.builder,
            LLVMConstInt(LLVMInt32Type(), self.addr, 0),
            self.pc_ptr(),
        );

        // Update the instret counter.
        let instret = LLVMBuildLoad(self.builder, self.instret_ptr(), NONAME);
        let instret = LLVMBuildAdd(
            self.builder,
            instret,
            LLVMConstInt(LLVMTypeOf(instret), 1, 0),
            NONAME,
        );
        LLVMBuildStore(self.builder, instret, self.instret_ptr());

        match self.inst {
            riscv::Format::Bimm12hiBimm12loRs1Rs2(x) => self.emit_bimm12hi_bimm12lo_rs1_rs2(x),
            riscv::Format::Imm12RdRs1(x) => self.emit_imm12_rd_rs1(x),
            riscv::Format::Imm12hiImm12loRs1Rs2(x) => self.emit_imm12hi_imm12lo_rs1_rs2(x),
            riscv::Format::Imm20Rd(x) => self.emit_imm20_rd(x),
            riscv::Format::Jimm20Rd(x) => self.emit_jimm20_rd(x),
            riscv::Format::RdRs1Rs2(x) => self.emit_rd_rs1_rs2(x),
            riscv::Format::RdRs1Shamt(x) => self.emit_rd_rs1_shamt(x),
            riscv::Format::Unit(x) => self.emit_unit(x),
            _ => Err(anyhow!("Unsupported instruction format")),
        }
        .with_context(|| format!("Unsupported instruction 0x{:x}: {}", self.addr, self.inst))
    }

    unsafe fn emit_bimm12hi_bimm12lo_rs1_rs2(
        &self,
        data: riscv::FormatBimm12hiBimm12loRs1Rs2,
    ) -> Result<()> {
        let target = (self.addr as i64).wrapping_add(data.bimm() as i64) as u64;
        trace!("{} x{}, x{}, 0x{:x}", data.op, data.rs1, data.rs2, target);
        let rs1 = self.read_reg(data.rs1);
        let rs2 = self.read_reg(data.rs2);
        let name = format!("{}_x{}_x{}\0", data.op, data.rs1, data.rs2);
        let name = name.as_ptr() as *const _;
        let predicate = match data.op {
            riscv::OpcodeBimm12hiBimm12loRs1Rs2::Beq => LLVMIntEQ,
            riscv::OpcodeBimm12hiBimm12loRs1Rs2::Bne => LLVMIntNE,
            riscv::OpcodeBimm12hiBimm12loRs1Rs2::Blt => LLVMIntSLT,
            riscv::OpcodeBimm12hiBimm12loRs1Rs2::Bge => LLVMIntSGE,
            riscv::OpcodeBimm12hiBimm12loRs1Rs2::Bltu => LLVMIntULT,
            riscv::OpcodeBimm12hiBimm12loRs1Rs2::Bgeu => LLVMIntUGE,
        };
        let cmp = LLVMBuildICmp(self.builder, predicate, rs1, rs2, name);
        let bb =
            LLVMCreateBasicBlockInContext(self.section.engine.context, b"\0".as_ptr() as *const _);
        LLVMInsertExistingBasicBlockAfterInsertBlock(self.builder, bb);
        LLVMBuildCondBr(self.builder, cmp, self.section.elf.inst_bbs[&target], bb);
        LLVMPositionBuilderAtEnd(self.builder, bb);
        Ok(())
    }

    unsafe fn emit_imm12hi_imm12lo_rs1_rs2(
        &self,
        data: riscv::FormatImm12hiImm12loRs1Rs2,
    ) -> Result<()> {
        let imm = data.imm();
        trace!("{} x{}, x{}, 0x{:x}", data.op, data.rs1, data.rs2, imm);
        let rs1 = self.read_reg(data.rs1);
        let rs2 = self.read_reg(data.rs2);
        let imm = LLVMConstInt(LLVMInt32Type(), (imm as i64) as u64, 0);
        // let name = format!("{}\0", data.op);
        // let name = name.as_ptr() as *const _;
        match data.op {
            riscv::OpcodeImm12hiImm12loRs1Rs2::Sb => self.emit_store(rs1, imm, rs2, 0),
            riscv::OpcodeImm12hiImm12loRs1Rs2::Sh => self.emit_store(rs1, imm, rs2, 1),
            riscv::OpcodeImm12hiImm12loRs1Rs2::Sw => self.emit_store(rs1, imm, rs2, 2),
            _ => bail!("Unsupported opcode {}", data.op),
        };
        Ok(())
    }

    unsafe fn emit_imm12_rd_rs1(&self, data: riscv::FormatImm12RdRs1) -> Result<()> {
        let imm = data.imm();
        trace!("{} x{} = x{}, 0x{:x}", data.op, data.rd, data.rs1, imm);

        // Handle CSR instructions, since we don't want to gratuitously read
        // registers.
        match data.op {
            riscv::OpcodeImm12RdRs1::Csrrwi
            | riscv::OpcodeImm12RdRs1::Csrrsi
            | riscv::OpcodeImm12RdRs1::Csrrci => return self.emit_csr_imm(data),
            riscv::OpcodeImm12RdRs1::Csrrw
            | riscv::OpcodeImm12RdRs1::Csrrs
            | riscv::OpcodeImm12RdRs1::Csrrc => return self.emit_csr_reg(data),
            _ => (),
        }

        let rs1 = self.read_reg(data.rs1);
        let imm = LLVMConstInt(LLVMInt32Type(), (imm as i64) as u64, 0);
        let name = format!("{}\0", data.op);
        let name = name.as_ptr() as *const _;
        let value = match data.op {
            riscv::OpcodeImm12RdRs1::Addi => LLVMBuildAdd(self.builder, rs1, imm, name),
            riscv::OpcodeImm12RdRs1::Slti => LLVMBuildZExt(
                self.builder,
                LLVMBuildICmp(self.builder, LLVMIntSLT, rs1, imm, NONAME),
                LLVMInt32Type(),
                name,
            ),
            riscv::OpcodeImm12RdRs1::Sltiu => LLVMBuildZExt(
                self.builder,
                LLVMBuildICmp(self.builder, LLVMIntULT, rs1, imm, NONAME),
                LLVMInt32Type(),
                name,
            ),
            riscv::OpcodeImm12RdRs1::Andi => LLVMBuildAnd(self.builder, rs1, imm, name),
            riscv::OpcodeImm12RdRs1::Ori => LLVMBuildOr(self.builder, rs1, imm, name),
            riscv::OpcodeImm12RdRs1::Xori => LLVMBuildXor(self.builder, rs1, imm, name),
            riscv::OpcodeImm12RdRs1::Lb => self.emit_load(rs1, imm, 0, false),
            riscv::OpcodeImm12RdRs1::Lh => self.emit_load(rs1, imm, 1, false),
            riscv::OpcodeImm12RdRs1::Lw => self.emit_load(rs1, imm, 2, false),
            riscv::OpcodeImm12RdRs1::Lbu => self.emit_load(rs1, imm, 0, true),
            riscv::OpcodeImm12RdRs1::Lhu => self.emit_load(rs1, imm, 1, true),
            riscv::OpcodeImm12RdRs1::Csrrw
            | riscv::OpcodeImm12RdRs1::Csrrs
            | riscv::OpcodeImm12RdRs1::Csrrc
            | riscv::OpcodeImm12RdRs1::Csrrwi
            | riscv::OpcodeImm12RdRs1::Csrrsi
            | riscv::OpcodeImm12RdRs1::Csrrci => unreachable!("handled above"),
            riscv::OpcodeImm12RdRs1::Jalr => {
                // Compute the branch target address.
                let target = LLVMBuildAdd(self.builder, rs1, imm, name);

                // Write the link register.
                self.write_reg(
                    data.rd,
                    LLVMConstInt(LLVMInt32Type(), (self.addr + 4) as u64, 0),
                );

                // Create a basic block where we land in case of an unpredicted
                // branch target (not in the `target_addrs` set).
                let bb = LLVMCreateBasicBlockInContext(
                    self.section.engine.context,
                    b"\0".as_ptr() as *const _,
                );
                LLVMInsertExistingBasicBlockAfterInsertBlock(self.builder, bb);

                // Emit the switch statement with all branch targets.
                let target_addrs = &self.section.elf.target_addrs;
                let sw = LLVMBuildSwitch(self.builder, target, bb, target_addrs.len() as u32);
                for &addr in target_addrs {
                    LLVMAddCase(
                        sw,
                        LLVMConstInt(LLVMInt32Type(), addr as u64, 0),
                        self.section.elf.inst_bbs[&addr],
                    );
                }

                // Emit the illegal branch code.
                LLVMPositionBuilderAtEnd(self.builder, bb);
                self.section.emit_branch_abort(self.addr, target);
                return Ok(()); // we have already written the link register
            }
            _ => bail!("Unsupported opcode {}", data.op),
        };
        self.write_reg(data.rd, value);
        Ok(())
    }

    unsafe fn emit_imm20_rd(&self, data: riscv::FormatImm20Rd) -> Result<()> {
        let imm = data.imm20 << 12;
        trace!("{} x{} = 0x{:x}", data.op, data.rd, imm);
        let value = match data.op {
            riscv::OpcodeImm20Rd::Auipc => LLVMConstInt(
                LLVMInt32Type(),
                (self.addr as u32).wrapping_add(imm) as u64,
                0,
            ),
            riscv::OpcodeImm20Rd::Lui => LLVMConstInt(LLVMInt32Type(), imm as u64, 0),
        };
        self.write_reg(data.rd, value);
        Ok(())
    }

    unsafe fn emit_jimm20_rd(&self, data: riscv::FormatJimm20Rd) -> Result<()> {
        match data.op {
            riscv::OpcodeJimm20Rd::Jal => {
                let target = (self.addr as i64).wrapping_add(data.jimm() as i64) as u64;
                trace!("jal x{}, 0x{:x}", data.rd, target);
                self.write_reg(
                    data.rd,
                    LLVMConstInt(LLVMInt32Type(), (self.addr + 4) as u64, 0),
                );
                LLVMBuildBr(self.builder, self.section.elf.inst_bbs[&target]);
                self.was_terminator.set(true);
                Ok(())
            }
        }
    }

    unsafe fn emit_rd_rs1_rs2(&self, data: riscv::FormatRdRs1Rs2) -> Result<()> {
        trace!("{} x{} = x{}, x{}", data.op, data.rd, data.rs1, data.rs2);
        let rs1 = self.read_reg(data.rs1);
        let rs2 = self.read_reg(data.rs2);
        let name = format!("{}\0", data.op);
        let name = name.as_ptr() as *const _;
        let value = match data.op {
            riscv::OpcodeRdRs1Rs2::Add => LLVMBuildAdd(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Sub => LLVMBuildSub(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::And => LLVMBuildAnd(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Or => LLVMBuildOr(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Xor => LLVMBuildXor(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Mul => LLVMBuildMul(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Div => LLVMBuildSDiv(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Divu => LLVMBuildUDiv(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Rem => LLVMBuildSRem(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Remu => LLVMBuildURem(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Slt => LLVMBuildZExt(
                self.builder,
                LLVMBuildICmp(self.builder, LLVMIntSLT, rs1, rs2, NONAME),
                LLVMInt32Type(),
                name,
            ),
            riscv::OpcodeRdRs1Rs2::Sltu => LLVMBuildZExt(
                self.builder,
                LLVMBuildICmp(self.builder, LLVMIntULT, rs1, rs2, NONAME),
                LLVMInt32Type(),
                name,
            ),
            riscv::OpcodeRdRs1Rs2::Sll => LLVMBuildShl(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Srl => LLVMBuildLShr(self.builder, rs1, rs2, name),
            riscv::OpcodeRdRs1Rs2::Sra => LLVMBuildAShr(self.builder, rs1, rs2, name),
            _ => bail!("Unsupported opcode {}", data.op),
        };
        self.write_reg(data.rd, value);
        Ok(())
    }

    unsafe fn emit_rd_rs1_shamt(&self, data: riscv::FormatRdRs1Shamt) -> Result<()> {
        trace!(
            "{} x{} = x{}, 0x{:x}",
            data.op,
            data.rd,
            data.rs1,
            data.shamt
        );
        let rs1 = self.read_reg(data.rs1);
        let shamt = LLVMConstInt(LLVMInt32Type(), data.shamt as u64, 0);
        let name = format!("{}\0", data.op);
        let name = name.as_ptr() as *const _;
        let value = match data.op {
            riscv::OpcodeRdRs1Shamt::Slli => LLVMBuildShl(self.builder, rs1, shamt, name),
            riscv::OpcodeRdRs1Shamt::Srli => LLVMBuildLShr(self.builder, rs1, shamt, name),
            riscv::OpcodeRdRs1Shamt::Srai => LLVMBuildAShr(self.builder, rs1, shamt, name),
            _ => bail!("Unsupported opcode {}", data.op),
        };
        self.write_reg(data.rd, value);
        Ok(())
    }

    unsafe fn emit_unit(&self, data: riscv::FormatUnit) -> Result<()> {
        trace!("{}", data.op,);
        match data.op {
            riscv::OpcodeUnit::Wfi => LLVMBuildRetVoid(self.builder),
            _ => bail!("Unsupported opcode {}", data.op),
        };
        Ok(())
    }

    unsafe fn emit_load(
        &self,
        base: LLVMValueRef,
        offset: LLVMValueRef,
        size: usize,
        sext: bool,
    ) -> LLVMValueRef {
        self.read_mem(LLVMBuildAdd(self.builder, base, offset, NONAME), size, sext)
    }

    unsafe fn emit_store(
        &self,
        base: LLVMValueRef,
        offset: LLVMValueRef,
        value: LLVMValueRef,
        size: usize,
    ) {
        self.write_mem(
            LLVMBuildAdd(self.builder, base, offset, NONAME),
            value,
            size,
        )
    }

    /// Emit the code to read-modify-write a CSR with an immediate rs1.
    unsafe fn emit_csr_imm(&self, data: riscv::FormatImm12RdRs1) -> Result<()> {
        self.emit_csr(data, LLVMConstInt(LLVMInt32Type(), data.rs1 as u64, 0))
    }

    /// Emit the code to read-modify-write a CSR with a register rs1.
    unsafe fn emit_csr_reg(&self, data: riscv::FormatImm12RdRs1) -> Result<()> {
        self.emit_csr(data, self.read_reg(data.rs1))
    }

    /// Emit the code to read-modify-write a CSR with an rs1 LLVM value.
    unsafe fn emit_csr(&self, data: riscv::FormatImm12RdRs1, value: LLVMValueRef) -> Result<()> {
        match data.op {
            riscv::OpcodeImm12RdRs1::Csrrw | riscv::OpcodeImm12RdRs1::Csrrwi if data.rd == 0 => {
                self.write_csr(data.imm12, value);
            }
            riscv::OpcodeImm12RdRs1::Csrrw | riscv::OpcodeImm12RdRs1::Csrrwi => {
                let prev = self.read_csr(data.imm12);
                self.write_csr(data.imm12, value);
                self.write_reg(data.rd, prev);
            }
            riscv::OpcodeImm12RdRs1::Csrrs
            | riscv::OpcodeImm12RdRs1::Csrrsi
            | riscv::OpcodeImm12RdRs1::Csrrc
            | riscv::OpcodeImm12RdRs1::Csrrci
                if data.rs1 == 0 =>
            {
                self.write_reg(data.rd, self.read_csr(data.imm12))
            }
            riscv::OpcodeImm12RdRs1::Csrrs | riscv::OpcodeImm12RdRs1::Csrrsi => {
                let prev = self.read_csr(data.imm12);
                let value = LLVMBuildOr(self.builder, prev, value, NONAME);
                self.write_csr(data.imm12, value);
                self.write_reg(data.rd, prev);
            }
            riscv::OpcodeImm12RdRs1::Csrrc | riscv::OpcodeImm12RdRs1::Csrrci => {
                let prev = self.read_csr(data.imm12);
                let value = LLVMBuildAnd(
                    self.builder,
                    prev,
                    LLVMBuildNot(self.builder, value, NONAME),
                    NONAME,
                );
                self.write_csr(data.imm12, value);
                self.write_reg(data.rd, prev);
            }
            _ => unreachable!("non-csr inst {}", data),
        }
        Ok(())
    }

    /// Emit the code necessary to load a value from memory.
    unsafe fn read_mem(&self, addr: LLVMValueRef, size: usize, sext: bool) -> LLVMValueRef {
        let value = LLVMBuildCall(
            self.builder,
            LLVMGetNamedFunction(
                self.section.engine.module,
                "banshee_load\0".as_ptr() as *const _,
            ),
            [
                self.section.state_ptr,
                addr,
                LLVMConstInt(LLVMInt8Type(), size as u64, 0),
            ]
            .as_mut_ptr(),
            3,
            NONAME,
        );
        if sext {
            let ty = LLVMIntType(8 << size);
            let value = LLVMBuildTrunc(self.builder, value, ty, NONAME);
            let value = LLVMBuildSExt(self.builder, value, LLVMInt32Type(), NONAME);
            value
        } else {
            value
        }
    }

    /// Emit the code necessary to store a value to memory.
    unsafe fn write_mem(&self, addr: LLVMValueRef, value: LLVMValueRef, size: usize) {
        LLVMBuildCall(
            self.builder,
            LLVMGetNamedFunction(
                self.section.engine.module,
                "banshee_store\0".as_ptr() as *const _,
            ),
            [
                self.section.state_ptr,
                addr,
                value,
                LLVMConstInt(LLVMInt8Type(), size as u64, 0),
            ]
            .as_mut_ptr(),
            4,
            NONAME,
        );
    }

    /// Emit the code necessary to read a value from a register.
    unsafe fn read_reg(&self, rs: u32) -> LLVMValueRef {
        if rs == 0 {
            LLVMConstInt(LLVMInt32Type(), 0, 0)
        } else {
            let ptr = self.reg_ptr(rs);
            LLVMBuildLoad(self.builder, ptr, format!("x{}\0", rs).as_ptr() as *const _)
        }
    }

    /// Emit the code necessary to write a value to a register.
    unsafe fn write_reg(&self, rd: u32, data: LLVMValueRef) {
        if rd != 0 {
            let ptr = self.reg_ptr(rd);
            LLVMBuildStore(self.builder, data, ptr);
        }
    }

    /// Emit the code necessary to read a value from a register.
    unsafe fn read_csr(&self, csr: u32) -> LLVMValueRef {
        LLVMBuildCall(
            self.builder,
            LLVMGetNamedFunction(
                self.section.engine.module,
                "banshee_csr_read\0".as_ptr() as *const _,
            ),
            [
                self.section.state_ptr,
                LLVMConstInt(LLVMInt16Type(), csr as u64, 0),
            ]
            .as_mut_ptr(),
            2,
            NONAME,
        )
    }

    /// Emit the code necessary to write a value to a register.
    unsafe fn write_csr(&self, csr: u32, data: LLVMValueRef) {
        LLVMBuildCall(
            self.builder,
            LLVMGetNamedFunction(
                self.section.engine.module,
                "banshee_csr_write\0".as_ptr() as *const _,
            ),
            [
                self.section.state_ptr,
                LLVMConstInt(LLVMInt16Type(), csr as u64, 0),
                data,
            ]
            .as_mut_ptr(),
            3,
            NONAME,
        );
    }

    unsafe fn reg_ptr(&self, r: u32) -> LLVMValueRef {
        assert!(r < 32);
        LLVMBuildGEP(
            self.builder,
            self.section.state_ptr,
            [
                LLVMConstInt(LLVMInt32Type(), 0, 0),
                LLVMConstInt(LLVMInt32Type(), 1, 0),
                LLVMConstInt(LLVMInt32Type(), r as u64, 0),
            ]
            .as_mut_ptr(),
            3 as u32,
            format!("ptr_x{}\0", r).as_ptr() as *const _,
        )
    }

    unsafe fn pc_ptr(&self) -> LLVMValueRef {
        LLVMBuildGEP(
            self.builder,
            self.section.state_ptr,
            [
                LLVMConstInt(LLVMInt32Type(), 0, 0),
                LLVMConstInt(LLVMInt32Type(), 2, 0),
            ]
            .as_mut_ptr(),
            2 as u32,
            format!("ptr_pc\0").as_ptr() as *const _,
        )
    }

    unsafe fn instret_ptr(&self) -> LLVMValueRef {
        LLVMBuildGEP(
            self.builder,
            self.section.state_ptr,
            [
                LLVMConstInt(LLVMInt32Type(), 0, 0),
                LLVMConstInt(LLVMInt32Type(), 3, 0),
            ]
            .as_mut_ptr(),
            2 as u32,
            format!("ptr_pc\0").as_ptr() as *const _,
        )
    }
}