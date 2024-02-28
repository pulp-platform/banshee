// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Generic, memory-mapped peripherals implemented using runtime callbacks.
use crate::configuration::Callback;
use crate::Cpu;
use ndarray::{s, Array1, Array2, Array3};
use std::sync::atomic::{AtomicU32, Ordering};
use PeriphReq::{Load, Store};

/// Reference held by execution engine, referencing each peripheral instance in each cluster
pub struct Peripherals {
    peripherals: Vec<Box<dyn Peripheral>>,
    cluster_peripherals: Vec<Vec<(u32, usize)>>,
}

unsafe impl Sync for Peripherals {}

impl Peripherals {
    pub fn new() -> Self {
        Self {
            peripherals: get_peripheral_types(),
            cluster_peripherals: Default::default(),
        }
    }

    pub fn add_cluster(&mut self, callbacks: &Vec<Callback>) {
        self.cluster_peripherals.push(
            callbacks
                .iter()
                .map(|x| {
                    (
                        x.size,
                        self.peripherals
                            .iter()
                            .position(|p| x.name.eq(&p.get_name()))
                            .expect(&format!("Undefined peripheral type: {}", x.name)[..]),
                    )
                })
                .collect(),
        );
    }

    pub fn load(&self, cpu: &Cpu, cluster_id: usize, addr: u32, size: u8) -> u32 {
        self.load_store(cpu, cluster_id, addr, size, Load)
    }

    pub fn store(&self, cpu: &Cpu, cluster_id: usize, addr: u32, value: u32, mask: u32, size: u8) {
        self.load_store(cpu, cluster_id, addr, size, Store(value, mask));
    }

    fn load_store(
        &self,
        cpu: &Cpu,
        cluster_id: usize,
        mut addr: u32,
        size: u8,
        req: PeriphReq,
    ) -> u32 {
        for i in &self.cluster_peripherals[cluster_id] {
            if addr < i.0 {
                return match req {
                    Load => {
                        trace!(
                            "Periph load from {}: cluster_id {}, offs 0x{:x}, size {}",
                            self.peripherals[i.1].get_name(),
                            cluster_id,
                            addr,
                            size
                        );
                        self.peripherals[i.1].load(cpu, addr, size)
                    }
                    Store(val, mask) => {
                        trace!(
                            "Periph store to {}: cluster_id {}, offs 0x{:x}, size {}, mask 0x{:x}, val {}",
                            self.peripherals[i.1].get_name(),
                            cluster_id,
                            addr,
                            size,
                            mask,
                            val
                        );
                        self.peripherals[i.1].store(cpu, addr, val, mask, size);
                        0
                    }
                };
            }
            addr = addr - i.0;
        }
        // Handle unmapped accesses: have no side effect on peripherals
        // TODO: should we trigger an error-response-like exception here?
        match req {
            Load => trace!(
                "Unmapped periph load: cluster_id {}, addr {}, size {}",
                cluster_id,
                addr,
                size
            ),
            Store(val, mask) => trace!(
                "Unmapped periph store: cluster_id {}, addr {}, size {}, mask {}, val {}",
                cluster_id,
                addr,
                size,
                mask,
                val
            ),
        }
        0
    }
}

enum PeriphReq {
    Load,
    Store(u32, u32),
}

/// Trait representing a peripheral
pub trait Peripheral {
    /// should return the same name as in the config file
    fn get_name(&self) -> &'static str;
    /// store instruction
    fn store(&self, cpu: &Cpu, addr: u32, value: u32, mask: u32, size: u8);
    /// load instruction
    fn load(&self, cpu: &Cpu, addr: u32, size: u8) -> u32;
}

/// Function called by the cpu to get the peripheral types. This function should
/// return a vector containing an instance of each available peripherable type.
/// To add a new peripheral type, declare it below and add it here.
pub fn get_peripheral_types() -> Vec<Box<dyn Peripheral>> {
    vec![
        Box::new(Semaphores::default()),
        Box::new(Fence::default()),
        Box::new(ZeroMemory::default()),
        Box::new(MemPoolDMA::default()),
        Box::new(MemPoolITA::default()),
    ]
}

#[derive(Default)]
struct Fence {
    set: AtomicU32,
    current: AtomicU32,
}

impl Peripheral for Fence {
    fn get_name(&self) -> &'static str {
        "fence"
    }

    fn store(&self, _cpu: &Cpu, addr: u32, val: u32, _mask: u32, _: u8) {
        match addr {
            0x0 => self.set.store(val, Ordering::SeqCst),
            _ => self.current.store(val, Ordering::SeqCst),
        }
    }

    fn load(&self, _cpu: &Cpu, _: u32, _: u8) -> u32 {
        self.current.fetch_add(1, Ordering::SeqCst);
        while self.set.load(Ordering::SeqCst) != self.current.load(Ordering::SeqCst) {}
        0
    }
}

#[derive(Default)]
struct Semaphores {
    empty_count: AtomicU32,
    full_count: AtomicU32,
    use_queue: AtomicU32,
}

impl Peripheral for Semaphores {
    fn get_name(&self) -> &'static str {
        "semaphores"
    }

    fn store(&self, _cpu: &Cpu, addr: u32, val: u32, _mask: u32, _: u8) {
        match addr {
            0x0 => self.empty_count.store(val, Ordering::SeqCst),
            0x4 => {
                self.empty_count.fetch_add(val, Ordering::SeqCst);
            }
            0x8 => {
                while self
                    .empty_count
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x >= val {
                            Some(x - val)
                        } else {
                            None
                        }
                    })
                    .is_err()
                {}
            }
            0xc => self.full_count.store(val, Ordering::SeqCst),
            0x10 => {
                self.full_count.fetch_add(val, Ordering::SeqCst);
            }
            0x14 => {
                while self
                    .full_count
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x >= val {
                            Some(x - val)
                        } else {
                            None
                        }
                    })
                    .is_err()
                {}
            }
            0x18 => self.use_queue.store(val, Ordering::SeqCst),
            0x1c => {
                self.use_queue.fetch_add(val, Ordering::SeqCst);
            }
            _ => {
                while self
                    .use_queue
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x >= val {
                            Some(x - val)
                        } else {
                            None
                        }
                    })
                    .is_err()
                {}
            }
        }
    }

    fn load(&self, _cpu: &Cpu, _: u32, _: u8) -> u32 {
        0
    }
}

#[derive(Default)]
struct ZeroMemory {}

impl Peripheral for ZeroMemory {
    fn get_name(&self) -> &'static str {
        "zero-memory"
    }

    fn store(&self, _cpu: &Cpu, _: u32, _: u32, _: u32, _: u8) {}

    fn load(&self, _cpu: &Cpu, _: u32, _: u8) -> u32 {
        0
    }
}

#[derive(Default)]
struct MemPoolDMA {
    src_addr: AtomicU32,
    dst_addr: AtomicU32,
    num_bytes: AtomicU32,
    conf: AtomicU32,
    status: AtomicU32,
    next_id: AtomicU32,
    done: AtomicU32,
}

impl Peripheral for MemPoolDMA {
    /// should return the same name as in the config file
    fn get_name(&self) -> &'static str {
        "mempool-dma"
    }
    /// store instruction
    fn store(&self, _cpu: &Cpu, addr: u32, value: u32, _mask: u32, _size: u8) {
        match addr {
            0x00 => self.src_addr.store(value, Ordering::SeqCst),
            0x04 => self.dst_addr.store(value, Ordering::SeqCst),
            0x08 => self.num_bytes.store(value, Ordering::SeqCst),
            0x0C => self.conf.store(value, Ordering::SeqCst),
            0x10 => (), /* status: Write has no effect */
            0x14 => (), /* next_id: Write has no effect */
            0x18 => (), /* done: Write has no effect */
            _ => unimplemented!(),
        }
        self.done.store(0, Ordering::SeqCst);
    }
    /// load instruction
    fn load(&self, cpu: &Cpu, addr: u32, _size: u8) -> u32 {
        match addr {
            0x00 => self.src_addr.load(Ordering::SeqCst),
            0x04 => self.dst_addr.load(Ordering::SeqCst),
            0x08 => self.num_bytes.load(Ordering::SeqCst),
            0x0C => self.conf.load(Ordering::SeqCst),
            0x10 => self.status.load(Ordering::SeqCst),
            0x14 => {
                cpu.binary_memcpy(
                    self.dst_addr.load(Ordering::SeqCst),
                    self.src_addr.load(Ordering::SeqCst),
                    self.num_bytes.load(Ordering::SeqCst),
                );
                self.done.store(1, Ordering::SeqCst);
                self.next_id.load(Ordering::SeqCst)
            }
            0x18 => self.done.load(Ordering::SeqCst),
            _ => unimplemented!(),
        }
    }
}

#[derive(Default)]
struct MemPoolITA {
    state: [AtomicU32; 4],
    start_addr: [AtomicU32; 4],
    out_addr: [AtomicU32; 4],
    rqs_addr: [AtomicU32; 4],
    seq_len: [AtomicU32; 4],
    emb_len: [AtomicU32; 4],
    proj_len: [AtomicU32; 4],
}
impl Peripheral for MemPoolITA {
    /// should return the same name as in the config file
    fn get_name(&self) -> &'static str {
        "mempool-ita"
    }

    /// store instruction
    fn store(&self, cpu: &Cpu, addr: u32, value: u32, _mask: u32, _size: u8) {
        let i = addr as usize / 0x30;
        let addr = addr as usize % 0x30;

        match addr {
            0x00 => unsafe {
                self.state[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store state 0x{:02x}",
                    i, &cpu.hartid, value
                );
                // Out addresses are currently hardcoded in ITA
                let mut return_value = value;
                if value & 0x1 == 1 {
                    // Start ITA
                    self.run_ita(
                        cpu,
                        self.start_addr[i].load(Ordering::SeqCst),
                        // All ITA cores fetch the Q and K vector always from the address specified to core 0
                        self.start_addr[0].load(Ordering::SeqCst),
                        self.out_addr[i].load(Ordering::SeqCst),
                        self.rqs_addr[i].load(Ordering::SeqCst),
                        self.seq_len[i].load(Ordering::SeqCst),
                        self.emb_len[i].load(Ordering::SeqCst),
                        self.proj_len[i].load(Ordering::SeqCst),
                        16,
                    );
                    // Set busy flag
                    return_value |= 0x2;
                    // Clear start flag
                    return_value &= !0x1;

                    self.state[i].store(return_value, Ordering::SeqCst);
                    info!("[ITA {}, CPU {}] Done.", i, &cpu.hartid);
                    info!(
                        "[ITA {}, CPU {}] Store state 0x{:02x}",
                        i, &cpu.hartid, return_value
                    );
                }
            },
            0x04 => {
                self.start_addr[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store start address 0x{:08x}",
                    i, &cpu.hartid, value
                )
            }
            0x08 => {
                self.out_addr[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store out address 0x{:08x}",
                    i, &cpu.hartid, value
                )
            }
            0x0C => {
                self.rqs_addr[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store rqs_addr 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            0x10 => {
                self.seq_len[i].store(value, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store seq_len 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            0x14 => {
                self.emb_len[i].store(value, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store emb_len 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            0x18 => {
                self.proj_len[i].store(value, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store proj_len 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            _ => unimplemented!(),
        }
    }
    /// load instruction
    fn load(&self, cpu: &Cpu, addr: u32, _size: u8) -> u32 {
        let i = addr as usize / 0x30;
        let addr = addr as usize % 0x30;

        match addr {
            0x00 => {
                let state = self.state[i].load(Ordering::SeqCst) & 0xFF;
                info!(
                    "[ITA {}, CPU {}] Read state 0x{:02x}",
                    i, &cpu.hartid, state
                );

                let busy_flag = state & 0x02;

                // WIESEP: As we have no timing model, just set the done flag if the busy flag is set
                if busy_flag == 0x02 {
                    let mut new_state = state;
                    // Clear the busy flag
                    new_state &= !0x02;

                    // Set the done flag
                    new_state |= 0x4;
                    self.state[i].store(new_state, Ordering::SeqCst);
                    info!(
                        "[ITA {}, CPU {}] > ITA is done -> store state 0x{:02x}",
                        i, &cpu.hartid, new_state
                    );
                }
                state
            }
            0x04 => self.start_addr[i].load(Ordering::SeqCst),
            0x08 => self.out_addr[i].load(Ordering::SeqCst),
            0x0C => self.rqs_addr[i].load(Ordering::SeqCst),
            0x10 => self.seq_len[i].load(Ordering::SeqCst),
            0x14 => self.emb_len[i].load(Ordering::SeqCst),
            0x18 => self.proj_len[i].load(Ordering::SeqCst),
            _ => unimplemented!(),
        }
    }
}

impl MemPoolITA {
    fn transpose_2d_arrays<T>(array: &mut Array3<T>) -> Array3<T>
    where
        T: Clone,
    {
        return array.to_owned().permuted_axes([0, 2, 1]);
    }

    unsafe fn ita_load_2d_i32(cpu: &Cpu, data: &mut Array2<i32>, mut address: u32, m: u32, n: u32) {
        for j in 0..m {
            for i in 0..n {
                let word = cpu.binary_load(address, 2);
                data[[j as usize, i as usize]] = word as i32;
                address += 4;
            }
        }
    }

    unsafe fn ita_load_2d(
        cpu: &Cpu,
        data: &mut Array2<i8>,
        mut address: u32,
        m: u32,
        n: u32,
        splits: u32,
    ) {
        for split in 0..splits {
            for j in 0..m {
                for i in (0..n / splits).step_by(4) {
                    let word = cpu.binary_load(address, 2);
                    let elements = std::mem::transmute::<u32, [i8; 4]>(word);
                    for (offset, e) in elements.iter().enumerate() {
                        data[[j as usize, ((n / splits) * split + i) as usize + offset]] = *e;
                    }
                    address += 4;
                }
            }
        }
    }

    unsafe fn ita_load_3d(
        cpu: &Cpu,
        data: &mut Array3<i8>,
        mut address: u32,
        m: u32,
        n: u32,
        p: u32,
        splits: u32,
    ) {
        for split in 0..splits {
            for j in 0..m {
                for i in 0..n {
                    for h in (0..p / splits).step_by(4) {
                        let word = cpu.binary_load(address, 2);
                        let elements = std::mem::transmute::<u32, [i8; 4]>(word);
                        for (offset, e) in elements.iter().enumerate() {
                            data[[
                                j as usize,
                                i as usize,
                                ((p / splits) * split + h) as usize + offset,
                            ]] = *e;
                        }
                        address += 4;
                    }
                }
            }
        }
    }

    unsafe fn ita_store_2d(
        cpu: &Cpu,
        data: &Array2<i8>,
        address: u32,
        m: u32,
        n: u32,
        splits: u32,
    ) {
        let mut address_offset = 0;
        for split in 0..splits {
            for j in 0..m {
                for i in (0..n / splits).step_by(4) {
                    let mut elements = [0u8; 4];
                    for offset in 0..elements.len() {
                        elements[offset] =
                            data[[j as usize, ((n / splits) * split + i) as usize + offset]] as u8;
                    }
                    let word = u32::from_ne_bytes(elements);
                    cpu.binary_store(address + address_offset, word, u32::MAX, 2);
                    debug!(
                        "[ITA, CPU {}] Store OUT to 0x{:x}",
                        &cpu.hartid,
                        address + address_offset
                    );
                    address_offset += 4;
                }
            }
        }
    }

    unsafe fn run_ita(
        &self,
        cpu: &Cpu,
        start_address: u32,
        start_address_core0: u32,
        out_address: u32,
        rqs_address: u32,
        seq_len: u32,
        emb_len: u32,
        proj_len: u32,
        processing_engines: u32,
    ) {
        // Setup of matrices for query_projection_space_transformation and key_projection_space_transformation
        // Sequence of addresses are hardcoded
        let start = start_address;
        let w_o_addr = start;
        let w_v_addr = start + proj_len * emb_len;
        let w_k_addr = start + proj_len * emb_len * 2;
        let w_q_addr = start + proj_len * emb_len * 3 + seq_len * emb_len * 2;
        let b_o_addr = start + proj_len * emb_len * 4 + seq_len * emb_len * 2;
        let b_v_addr = start + proj_len * emb_len * 4 + seq_len * emb_len * 2 + emb_len * 4; // 32 bit biases
        let b_k_addr =
            start + proj_len * emb_len * 4 + seq_len * emb_len * 2 + emb_len * 4 + proj_len * 4; // 32 bit biases
        let b_q_addr =
            start + proj_len * emb_len * 4 + seq_len * emb_len * 2 + emb_len * 4 + proj_len * 8; // 32 bit biases

        let q_addr = start_address_core0 + proj_len * emb_len * 3;
        let k_addr = start_address_core0 + proj_len * emb_len * 3 + seq_len * emb_len;

        let mult_address = cpu.binary_load(rqs_address + 0x00, 2);
        let shift_address = cpu.binary_load(rqs_address + 0x04, 2);
        let add_address = cpu.binary_load(rqs_address + 0x08, 2);

        let rqs_mult_w1 = u32::to_ne_bytes(cpu.binary_load(mult_address + 0x00, 2));
        let rqs_mult_w2 = u32::to_ne_bytes(cpu.binary_load(mult_address + 0x04, 2));

        let rqs_mult: [u8; 6] = [
            rqs_mult_w1[0],
            rqs_mult_w1[1],
            rqs_mult_w1[2],
            rqs_mult_w1[3],
            rqs_mult_w2[0],
            rqs_mult_w2[1],
        ];

        let rqs_shift_w1 = u32::to_ne_bytes(cpu.binary_load(shift_address + 0x00, 2));
        let rqs_shift_w2: [u8; 4] = u32::to_ne_bytes(cpu.binary_load(shift_address + 0x04, 2));

        let rqs_shift: [u8; 6] = [
            rqs_shift_w1[0],
            rqs_shift_w1[1],
            rqs_shift_w1[2],
            rqs_shift_w1[3],
            rqs_shift_w2[0],
            rqs_shift_w2[1],
        ];

        let rqs_add: [i32; 6] = [
            cpu.binary_load(add_address + 0x00, 2) as i32,
            cpu.binary_load(add_address + 0x04, 2) as i32,
            cpu.binary_load(add_address + 0x08, 2) as i32,
            cpu.binary_load(add_address + 0x0C, 2) as i32,
            cpu.binary_load(add_address + 0x10, 2) as i32,
            cpu.binary_load(add_address + 0x14, 2) as i32,
        ];

        debug!("[ITA, CPU {}] w_o_addr 0x{:x}", &cpu.hartid, w_o_addr);
        debug!("[ITA, CPU {}] w_v_addr 0x{:x}", &cpu.hartid, w_v_addr);
        debug!("[ITA, CPU {}] w_k_addr 0x{:x}", &cpu.hartid, w_k_addr);
        debug!("[ITA, CPU {}] q_addr   0x{:x}", &cpu.hartid, q_addr);
        debug!("[ITA, CPU {}] k_addr   0x{:x}", &cpu.hartid, k_addr);
        debug!("[ITA, CPU {}] w_q_addr 0x{:x}", &cpu.hartid, w_q_addr);
        debug!("[ITA, CPU {}] b_o_addr 0x{:x}", &cpu.hartid, b_o_addr);
        debug!("[ITA, CPU {}] b_v_addr 0x{:x}", &cpu.hartid, b_v_addr);
        debug!("[ITA, CPU {}] b_k_addr 0x{:x}", &cpu.hartid, b_k_addr);
        debug!("[ITA, CPU {}] b_q_addr 0x{:x}", &cpu.hartid, b_q_addr);

        debug!(
            "[ITA, CPU {}] mult_address  0x{:x}",
            &cpu.hartid, mult_address
        );
        debug!(
            "[ITA, CPU {}] shift_address 0x{:x}",
            &cpu.hartid, shift_address
        );
        debug!(
            "[ITA, CPU {}] add_address   0x{:x}",
            &cpu.hartid, add_address
        );

        let split_e = emb_len / processing_engines;
        let split_p = proj_len / processing_engines;

        debug!(
            "[ITA, CPU {}] Start Address 0x{:x}, Out Address 0x{:x}",
            &cpu.hartid, start, out_address
        );
        debug!("[ITA, CPU {}] RQS Mult {:?}", &cpu.hartid, rqs_mult);
        debug!("[ITA, CPU {}] RQS Shift {:?}", &cpu.hartid, rqs_shift);
        debug!("[ITA, CPU {}] RQS Add {:?}", &cpu.hartid, rqs_add);
        debug!(
            "[ITA, CPU {}] S {:?}, E {:?}, P {:?}",
            &cpu.hartid, seq_len, emb_len, proj_len
        );
        debug!(
            "[ITA, CPU {}] Split E {:?}, Split P {:?}",
            &cpu.hartid, split_e, split_p
        );

        let mut q = Array2::<i8>::zeros((seq_len as usize, emb_len as usize));
        MemPoolITA::ita_load_2d(cpu, &mut q, q_addr, seq_len, emb_len, split_e);
        debug!("[ITA, CPU {}] q.shape: {:?}", &cpu.hartid, q.shape());
        debug!("[ITA, CPU {}] q: {}", &cpu.hartid, q);

        let mut w_q = Array3::<i8>::zeros((1, proj_len as usize, emb_len as usize));
        MemPoolITA::ita_load_3d(cpu, &mut w_q, w_q_addr, 1, proj_len, emb_len, split_e);
        w_q = MemPoolITA::transpose_2d_arrays(&mut w_q);
        debug!("[ITA, CPU {}] w_q.shape: {:?}", &cpu.hartid, w_q.shape());
        debug!("[ITA, CPU {}] w_q: {}", &cpu.hartid, w_q);

        let mut k = Array2::<i8>::zeros((seq_len as usize, emb_len as usize));
        MemPoolITA::ita_load_2d(cpu, &mut k, k_addr, seq_len, emb_len, split_e);
        debug!("[ITA, CPU {}] k.shape: {:?}", &cpu.hartid, k.shape());
        debug!("[ITA, CPU {}] k: {}", &cpu.hartid, k);

        let mut w_k = Array3::<i8>::zeros((1, proj_len as usize, emb_len as usize));
        MemPoolITA::ita_load_3d(cpu, &mut w_k, w_k_addr, 1, proj_len, emb_len, 1);
        w_k = MemPoolITA::transpose_2d_arrays(&mut w_k);
        debug!("[ITA, CPU {}] w_k.shape: {:?}", &cpu.hartid, w_k.shape());
        debug!("[ITA, CPU {}] w_k: {}", &cpu.hartid, w_k);

        // Setup of matrices for value_projection_space_transformation
        let mut b_v = Array2::<i32>::zeros((1, proj_len as usize));
        MemPoolITA::ita_load_2d_i32(cpu, &mut b_v, b_v_addr, 1, proj_len);
        debug!("[ITA, CPU {}] b_v.shape: {:?}", &cpu.hartid, b_v.shape());
        debug!("[ITA, CPU {}] b_v: {}", &cpu.hartid, b_v);

        let mut v = k.clone();
        let mut w_v = Array3::<i8>::zeros((1, proj_len as usize, emb_len as usize));
        MemPoolITA::ita_load_3d(cpu, &mut w_v, w_v_addr, 1, proj_len, emb_len, 1);
        w_v = MemPoolITA::transpose_2d_arrays(&mut w_v);
        debug!("[ITA, CPU {}] w_v.shape: {:?}", &cpu.hartid, w_v.shape());
        debug!("[ITA, CPU {}] w_v: {}", &cpu.hartid, w_v);

        let mut v_p = Array3::<i32>::zeros((1, seq_len as usize, proj_len as usize));

        // matrices in the query_projection_space_transformation
        let mut b_q = Array2::<i32>::zeros((1, proj_len as usize));
        MemPoolITA::ita_load_2d_i32(cpu, &mut b_q, b_q_addr, 1, proj_len);
        debug!("[ITA, CPU {}] b_q.shape: {:?}", &cpu.hartid, b_q.shape());
        debug!("[ITA, CPU {}] b_q: {}", &cpu.hartid, b_q);
        let mut q_p = Array3::<i32>::zeros((1, seq_len as usize, proj_len as usize));

        // matrices in the key_projection_space_transformation
        let mut b_k = Array2::<i32>::zeros((1, proj_len as usize));
        MemPoolITA::ita_load_2d_i32(cpu, &mut b_k, b_k_addr, 1, proj_len);
        debug!("[ITA, CPU {}] b_k.shape: {:?}", &cpu.hartid, b_k.shape());
        debug!("[ITA, CPU {}] b_k: {}", &cpu.hartid, b_k);

        let mut k_p = Array3::<i32>::zeros((1, seq_len as usize, proj_len as usize));

        // matrices in the streaming_partial_softmax
        let mut a_requant = Array3::<i8>::zeros((1, seq_len as usize, seq_len as usize));
        let mut a_partial_softmax = Array2::<i32>::zeros((seq_len as usize, seq_len as usize));

        // matrices in multi_head_computation
        let mut out = Array3::<i32>::zeros((1, seq_len as usize, emb_len as usize));
        let mut b_o = Array2::<i32>::zeros((1, emb_len as usize));
        MemPoolITA::ita_load_2d_i32(cpu, &mut b_o, b_o_addr, 1, emb_len);

        debug!("[ITA, CPU {}] b_o.shape: {:?}", &cpu.hartid, b_o.shape());
        debug!("[ITA, CPU {}] b_o: {}", &cpu.hartid, b_o);

        let mut w_o = Array3::<i8>::zeros((1, emb_len as usize, proj_len as usize));
        MemPoolITA::ita_load_3d(cpu, &mut w_o, w_o_addr, 1, emb_len, proj_len, 1);
        w_o = MemPoolITA::transpose_2d_arrays(&mut w_o);
        debug!("[ITA, CPU {}] w_o.shape: {:?}", &cpu.hartid, w_o.shape());
        debug!("[ITA, CPU {}] w_o: {}", &cpu.hartid, w_o);

        // query_projection_space_transformation
        MemPoolITA::projection_space_transformation(&mut q_p, &mut q, &mut w_q, &mut b_q, 1);
        // requantization of q_p
        let mut q_p_requant = Array3::<i8>::zeros((1, seq_len as usize, proj_len as usize));
        MemPoolITA::requantization_3d(
            &mut q_p,
            &mut q_p_requant,
            rqs_mult[0],
            rqs_shift[0],
            rqs_add[0],
        );
        debug!("[ITA, CPU {}] q_p_requant: {}", &cpu.hartid, q_p_requant);

        // key_projection_space_transformation
        MemPoolITA::projection_space_transformation(&mut k_p, &mut k, &mut w_k, &mut b_k, 1);
        // requantization of k_p
        let mut k_p_requant = Array3::<i8>::zeros((1, seq_len as usize, proj_len as usize));
        MemPoolITA::requantization_3d(
            &mut k_p,
            &mut k_p_requant,
            rqs_mult[1],
            rqs_shift[1],
            rqs_add[1],
        );
        debug!("[ITA, CPU {}] k_p_requant: {}", &cpu.hartid, k_p_requant);

        // query_key_correlation
        let mut qk = Array3::<i32>::zeros((1, seq_len as usize, seq_len as usize));
        MemPoolITA::query_key_correlation(&mut q_p_requant, &mut k_p_requant, &mut qk);
        // requantization of qk
        MemPoolITA::requantization_3d(
            &mut qk,
            &mut a_requant,
            rqs_mult[2],
            rqs_shift[2],
            rqs_add[2],
        );
        debug!("[ITA, CPU {}] a_requant: {}", &cpu.hartid, a_requant);

        // streaming_partial_softmax
        MemPoolITA::streaming_partial_softmax(
            &mut a_requant,
            &mut a_partial_softmax,
            seq_len,
            processing_engines,
        );

        // value_projection_space_transformation
        MemPoolITA::projection_space_transformation(&mut v_p, &mut v, &mut w_v, &mut b_v, 1);
        // requantization of v_p
        let mut v_p_requant = Array3::<i8>::zeros((1, seq_len as usize, proj_len as usize));
        MemPoolITA::requantization_3d(
            &mut v_p,
            &mut v_p_requant,
            rqs_mult[3],
            rqs_shift[3],
            rqs_add[3],
        );
        debug!("[ITA, CPU {}] v_p_requant: {}", &cpu.hartid, v_p_requant);

        // single_head_computation
        let mut o_softmax = Array3::<i32>::zeros((1, seq_len as usize, proj_len as usize));
        MemPoolITA::single_head_computation(
            &mut a_partial_softmax,
            &mut v_p_requant,
            &mut o_softmax,
        );
        // requantization of o_softmax
        let mut o_softmax_requant = Array3::<i8>::zeros((1, seq_len as usize, proj_len as usize));
        MemPoolITA::requantization_3d(
            &mut o_softmax,
            &mut o_softmax_requant,
            rqs_mult[4],
            rqs_shift[4],
            rqs_add[4],
        );
        debug!(
            "[ITA, CPU {}] o_softmax_requant: {}",
            &cpu.hartid, o_softmax_requant
        );

        // multi_head_computation
        MemPoolITA::multi_head_computation(&mut o_softmax_requant, &mut out, &mut w_o, &mut b_o, 1);
        // parallel requantization of out
        let mut out_requant = Array2::<i8>::zeros((seq_len as usize, emb_len as usize));
        MemPoolITA::parallel_requantize3d(
            &mut out,
            &mut out_requant,
            rqs_mult[5],
            rqs_shift[5],
            rqs_add[5],
        );
        debug!("[ITA, CPU {}] out_requant: {}", &cpu.hartid, out_requant);

        // Store the output
        MemPoolITA::ita_store_2d(cpu, &out_requant, out_address, seq_len, emb_len, 1);
    }

    fn requantize_row(element: i32, eps_mult: u8, right_shift: u8, add: i32) -> i8 {
        let mut shifted = ((element * (eps_mult as i32)) >> (right_shift as i32)) + (add as i32);

        // Perform rounding half away from zero
        if right_shift > 0
            && ((element * (eps_mult as i32)) >> ((right_shift - 1) as i32)) & 0x1 == 1
        {
            shifted = shifted.saturating_add(1);
        }
        if shifted > 127 {
            return 127;
        } else if shifted < -128 {
            return -128;
        } else {
            return shifted as i8;
        }
    }

    fn requantization_3d(
        m: &mut Array3<i32>,
        m_requant: &mut Array3<i8>,
        eps_mult: u8,
        right_shift: u8,
        add: i32,
    ) {
        // Loop over the number of heads
        for i in 0..m.shape()[0] {
            // Loop over the head dimension
            for j in 0..m.shape()[1] {
                // print the column of the head matrix
                let row = m.slice(s![i, j, ..]);
                // Iterate over the row and requantize it
                for k in 0..row.len() {
                    m_requant[[i, j, k]] =
                        MemPoolITA::requantize_row(row[k], eps_mult, right_shift, add);
                }
            }
        }
    }

    fn parallel_requantize3d(
        m: &mut Array3<i32>,
        m_requant: &mut Array2<i8>,
        eps_mult: u8,
        right_shift: u8,
        add: i32,
    ) {
        m_requant.fill(add as i8);
        for i in 0..m.shape()[0] {
            for j in 0..m.shape()[1] {
                let row = m.slice(s![i, j, ..]);
                for k in 0..row.len() {
                    let mut shifted = ((row[k] * (eps_mult as i32)) >> (right_shift as i32))
                        + m_requant[[i * m.shape()[1] + j, k]] as i32;

                    // Perform rounding half away from zero
                    if right_shift > 0
                        && ((row[k] * (eps_mult as i32)) >> ((right_shift - 1) as i32)) & 0x1 == 1
                    {
                        shifted = shifted.saturating_add(1);
                    }
                    m_requant[[i * m.shape()[1] + j, k]] =
                        MemPoolITA::requantize_row(shifted, 1, 0, 0);
                }
            }
        }
    }

    fn projection_space_transformation(
        p: &mut Array3<i32>,
        m: &mut Array2<i8>,
        w: &mut Array3<i8>,
        b: &mut Array2<i32>,
        bias: u8,
    ) {
        info!("===================== Projection Space Transformation =====================");
        info!("p shape: {:?}", p.shape());
        info!("m shape: {:?}", m.shape());
        info!("w: {:?}", w.shape());
        info!("b: {:?}", b.shape());

        // Calculate p[h] = m * W[h] + b[h] for each head h

        let d1 = m.shape();
        let d2 = w.shape();

        assert_eq!(d1[1], d2[1], "Matrices dimensions don't match");

        for i in 0..p.shape()[0] {
            let slice_a = m.map(|x| *x as i32);
            let slice_b = w.slice(s![i, .., ..]).map(|x| *x as i32);
            let slice_c = b.slice(s![i, ..]).map(|x| *x);
            let slice_c = slice_c.broadcast((d1[0], d2[2])).unwrap().map(|x| *x);
            let mut mult_a_b = slice_a.dot(&slice_b);

            if bias == 1 {
                mult_a_b = mult_a_b + slice_c;
            }

            p.slice_mut(s![i, .., ..]).assign(&mult_a_b);
        }
    }

    fn query_key_correlation(
        qp_requant: &mut Array3<i8>,
        kp_requant: &mut Array3<i8>,
        qk: &mut Array3<i32>,
    ) {
        info!("===================== Query Key Correlation =====================");
        info!("qp_requant shape: {:?}", qp_requant.shape());
        info!("kp_requant shape: {:?}", kp_requant.shape());
        info!("qk shape: {:?}", qk.shape());

        let d1 = qp_requant.shape();
        let d2 = kp_requant.shape();

        assert_eq!(d1[2], d2[2], "Matrices dimensions don't match");

        // Calculate qk[h] = qp_requant[h] * kp_requant[h].T for each head h
        let kp_requant_transposed = MemPoolITA::transpose_2d_arrays(kp_requant);

        for i in 0..qk.shape()[0] {
            let slice_a = qp_requant.slice(s![i, .., ..]).map(|x| *x as i32);
            let slice_b = kp_requant_transposed
                .slice(s![i, .., ..])
                .map(|x| *x as i32);
            let mult_a_b = slice_a.dot(&slice_b);

            qk.slice_mut(s![i, .., ..]).assign(&mult_a_b);
        }
    }

    //Compute the approximated softmax function.
    fn streaming_partial_softmax(
        a_requant: &mut Array3<i8>,
        a_partial_softmax: &mut Array2<i32>,
        seq_len: u32,
        processing_engines: u32,
    ) {
        // let log2e: f64 = f64::log2(f64::exp(1.0));
        // let b = 8;
        // let eps_x = b as f64 / (2.0f64.powi(b) * log2e);
        let mut exp_partial_sum = Array1::<i32>::zeros(seq_len as usize);
        let mut max = Array1::<i8>::zeros(seq_len as usize);
        let mut current_max = Array1::<i8>::zeros(seq_len as usize);
        let _processing_engines = processing_engines as usize;
        let groups = seq_len as usize / _processing_engines;

        for i in 0..groups {
            let a_requant_slice = a_requant.slice_mut(s![
                0,
                ..,
                i * _processing_engines..(i + 1) * _processing_engines
            ]);

            for n in 0..a_requant_slice.nrows() {
                current_max[[n]] = a_requant_slice.row(n).iter().copied().max().unwrap() as i8;
            }

            for j in 0..seq_len {
                let mut shift_sum: u8;
                if i == 0 || current_max[j as usize] > max[[j as usize]] {
                    if i == 0 {
                        shift_sum = 0;
                    } else {
                        let shift_int =
                            (current_max[j as usize] as i32) - (max[[j as usize]] as i32);
                        shift_sum = (shift_int / 32) as u8;

                        if shift_int % 32 >= 16 {
                            shift_sum += 1;
                        }
                    }
                    max[j as usize] = current_max[j as usize];
                } else {
                    shift_sum = 0;
                }

                let qb = a_requant
                    .slice_mut(s![
                        0,
                        ..,
                        i * _processing_engines..(i + 1) * _processing_engines
                    ])
                    .mapv(|x| x as i32 - max[[j as usize]] as i32);

                let mut qexp = 0;
                for k in 0..qb.ncols() {
                    let mut shift = (-qb[[j as usize, k]]) as i32 / 32;
                    let shift_int = (-qb[[j as usize, k]]) as i32;

                    if shift_int % 32 >= 16 {
                        shift += 1;
                    }

                    qexp += (2_u32.pow(10) >> shift as i32) as i32;
                }

                exp_partial_sum[[j as usize]] =
                    (exp_partial_sum[[j as usize]] >> shift_sum as i32) + qexp;
            }
        }
        for j in 0..seq_len {
            let factor =
                ((2.0f64.powi(7) - 1.0) * 2.0f64.powi(10)) as i32 / exp_partial_sum[j as usize];
            for k in 0..seq_len {
                let mut shift = (((max[j as usize] as i32)
                    - (a_requant[[0, j as usize, k as usize]] as i32))
                    / 32) as i32;
                let shift_int =
                    (max[j as usize] as i32) - (a_requant[[0, j as usize, k as usize]] as i32);
                if shift_int % 32 >= 16 {
                    shift += 1;
                }
                a_partial_softmax[[j as usize, k as usize]] =
                    (factor as i32) / 2.0f64.powi(shift) as i32;
            }
        }
    }

    fn single_head_computation(
        a_partial_softmax: &mut Array2<i32>,
        vp_requant: &mut Array3<i8>,
        o_softmax: &mut Array3<i32>,
    ) {
        // Loop over the number of heads
        for i in 0..o_softmax.shape()[0] {
            // Loop over the number of queries
            for j in 0..o_softmax.shape()[1] {
                // Loop over the number of keys
                for k in 0..o_softmax.shape()[2] {
                    o_softmax[[i, j, k]] = 0;
                    // Loop over the number of features
                    for l in 0..o_softmax.shape()[1] {
                        o_softmax[[i, j, k]] +=
                            a_partial_softmax[[j, l]] as i32 * vp_requant[[i, l, k]] as i32;
                    }
                }
            }
        }
    }

    fn multi_head_computation(
        o_softmax_requant: &mut Array3<i8>,
        out: &mut Array3<i32>,
        w_o: &mut Array3<i8>,
        b_o: &mut Array2<i32>,
        bias: u8,
    ) {
        info!("===================== Multi Head Computation =====================");
        info!("o_softmax_requant shape: {:?}", o_softmax_requant.shape());
        info!("out shape: {:?}", out.shape());
        info!("w_o shape: {:?}", w_o.shape());
        info!("b_o shape: {:?}", b_o.shape());

        let d1 = o_softmax_requant.shape();
        let d2 = w_o.shape();

        assert_eq!(d1[2], d2[1], "Matrices dimensions don't match");

        // Calculate out[h] = o_softmax_requant[h] * W_o[h] + b_o[h] for each head h
        for i in 0..out.shape()[0] {
            let slice_a = o_softmax_requant.slice(s![i, .., ..]).map(|x| *x as i32);
            let slice_b = w_o.slice(s![i, .., ..]).map(|x| *x as i32);
            let slice_c = b_o.slice(s![i, ..]).map(|x| *x);
            let slice_c = slice_c.broadcast((d1[0], d2[2])).unwrap().map(|x| *x);
            let mut mult_a_b = slice_a.dot(&slice_b);

            if bias == 1 {
                mult_a_b = mult_a_b + slice_c;
            }

            out.slice_mut(s![i, .., ..]).assign(&mult_a_b);
        }
    }
}
