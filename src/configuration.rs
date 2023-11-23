// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Configuration code to describe the simulated system architecture
//!
//! Some care is required when changing this structure, as it requires the
//! `engine.rs` and the `tran.rs` to be adapted accordingly.

use std::io::prelude::*;

/// A struct to store the whole system configuration
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Configuration {
    #[serde(default)]
    pub architecture: Architecture,
    #[serde(default)]
    pub bootrom: MemoryCallback,
    #[serde(default)]
    pub memory: Memories,
    #[serde(default)]
    pub address: Address,
    #[serde(default)]
    pub inst_latency: std::collections::HashMap<String, u64>,
    #[serde(default)]
    pub ssr: Ssr,
    #[serde(default)]
    pub interrupt_latency: u32,
}

impl Default for Configuration {
    fn default() -> Configuration {
        Configuration {
            architecture: Default::default(),
            bootrom: Default::default(),
            memory: Default::default(),
            address: Default::default(),
            inst_latency: Default::default(),
            ssr: Default::default(),
            interrupt_latency: 10,
        }
    }
}

impl std::fmt::Display for Configuration {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_yaml::to_string(self).unwrap())
    }
}

impl Configuration {
    pub fn new(num_clusters: usize, num_cores: usize, base_hartid: usize) -> Self {
        Self {
            architecture: Architecture::new(num_clusters, num_cores, base_hartid),
            bootrom: Default::default(),
            memory: Default::default(),
            address: Default::default(),
            inst_latency: Default::default(),
            ssr: Default::default(),
            interrupt_latency: 10,
        }
    }
    /// Parse a json/yaml file into a `Configuration` struct
    pub fn parse(
        name: &str,
        num_clusters: usize,
        has_num_clusters: bool,
        num_cores: usize,
        has_num_cores: bool,
        base_hartid: usize,
        has_base_hartid: bool,
    ) -> Configuration {
        let config: String = std::fs::read_to_string(name)
            .unwrap_or_else(|_| panic!("Could not open file {}", name))
            .parse()
            .unwrap_or_else(|_| panic!("Could not parse file {}", name));
        // Parse the configuration file based on it's name
        let mut config: Configuration = if name.to_lowercase().contains("json") {
            serde_json::from_str(&config).expect("Error while reading json")
        } else {
            serde_yaml::from_str(&config).expect("Error while reading yaml")
        };
        if has_num_cores {
            config.architecture.num_cores = num_cores;
        }
        if has_base_hartid {
            config.architecture.base_hartid = base_hartid;
        }
        if config.architecture.num_clusters == 0 || has_num_clusters {
            config.architecture.num_clusters = num_clusters;
        }
        config
    }

    /// Write the default `Configuration` struct into a json/yaml file
    pub fn print_default(name: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(name)?;
        let mut c = serde_json::to_value(Configuration::default()).unwrap();
        let i = c.get_mut("inst_latency").unwrap();
        let l = crate::riscv::Latency::default();
        *i = serde_json::to_value(l).unwrap();
        warn!("{:?}", c);

        if name.to_lowercase().contains("json") {
            f.write_all(serde_json::to_string_pretty(&c).unwrap().as_bytes())?;
        } else {
            f.write_all(serde_yaml::to_string(&c).unwrap().as_bytes())?;
        }
        Ok(())
    }
}

/// Holds all the memories in the hierarchy
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Memories {
    pub tcdm: Memory,
    pub dram: Memory,
    pub periphs: MemoryCallback,
    pub tcdm_alias: bool,
    pub tcdm_alias_start: u32,
}

impl Default for Memories {
    fn default() -> Memories {
        Memories {
            tcdm: Memory {
                start: 0x100000,
                size: 0x20000,
                offset: 0x40000,
                latency: 2,
            },
            dram: Memory {
                start: 0x80000000,
                size: 0x10000000,
                offset: 0x0,
                latency: 10,
            },
            periphs: MemoryCallback {
                start: 0x20000,
                size: 0x10000,
                offset: 0x40000,
                latency: 2,
                callbacks: vec![],
            },
            tcdm_alias: false,
            tcdm_alias_start: 0,
        }
    }
}

/// Description of a single memory hierarchy
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Memory {
    pub start: u32,
    pub size: u32,
    pub offset: u32,
    pub latency: u64,
}

impl Default for Memory {
    fn default() -> Memory {
        Memory {
            start: 0,
            size: u32::MAX,
            offset: 0,
            latency: 1,
        }
    }
}

/// Description of a single memory hierarchy with callback functions
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct MemoryCallback {
    pub start: u32,
    pub size: u32,
    pub offset: u32,
    pub latency: u64,
    pub callbacks: Vec<Callback>,
}

impl Default for MemoryCallback {
    fn default() -> Self {
        Self {
            start: 0,
            size: u32::MAX,
            offset: 0,
            latency: 1,
            callbacks: vec![],
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Callback {
    pub name: String,
    pub size: u32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct ExtTcdm {
    pub cluster: u32,
    pub start: u32,
}

/// Struct to configure specific addresses
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Address {
    pub tcdm_start: u32,
    pub tcdm_end: u32,
    pub nr_cores: u32,
    pub scratch_reg: u32,
    pub wakeup_reg: u32,
    pub barrier_reg: Barrier,
    pub cluster_base_hartid: u32,
    pub cluster_num: u32,
    pub cluster_id: u32,
    pub uart: u32,
    pub clint: u32,
    pub cl_clint: u32,
}

impl Default for Address {
    fn default() -> Address {
        Address {
            tcdm_start: 0x40000000,
            tcdm_end: 0x40000008,
            nr_cores: 0x40000010,
            scratch_reg: 0x40000020,
            wakeup_reg: 0x40000028,
            barrier_reg: Barrier::default(),
            cluster_base_hartid: 0x40000040,
            cluster_num: 0x40000048,
            cluster_id: 0x40000050,
            cl_clint: 0x40000060,
            uart: 0xF00B8000,
            clint: 0xFFFF0000,
        }
    }
}

/// Struct to configure SSRs
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Ssr {
    pub num_dm: usize,
}

impl Default for Ssr {
    fn default() -> Ssr {
        Ssr { num_dm: 3 }
    }
}

/// Description of the hierarchy
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Architecture {
    pub num_cores: usize,
    pub num_clusters: usize,
    pub base_hartid: usize,
}

impl Architecture {
    pub fn new(num_clusters: usize, num_cores: usize, base_hartid: usize) -> Self {
        Self {
            num_cores: num_cores,
            num_clusters: num_clusters,
            base_hartid: base_hartid,
        }
    }
}

impl Default for Architecture {
    fn default() -> Architecture {
        Architecture {
            num_cores: 0,
            num_clusters: 0,
            base_hartid: 0,
        }
    }
}

/// Description of barrier hierarchy for multiple clusters
#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Barrier {
    pub start: u32,
    pub offset: u32,
}

impl Default for Barrier {
    fn default() -> Barrier {
        Barrier {
            start: 0x40000038,
            offset: 0,
        }
    }
}
