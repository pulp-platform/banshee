// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

const FLEXFLOAT_DIR: &str = "../deps/flexfloat";

use cmake::Config;

fn main() {
    let src = ["../deps/flexfloat/src/flexfloat.c"];

    // Ensure that we rebuild whenever any of the input files changes.
    for f in &src {
        println!("cargo:rerun-if-changed={}", f);
    }

    let dst = Config::new(FLEXFLOAT_DIR)
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .no_build_target(true)
        .very_verbose(true)
        .build();

    println!("cargo:rustc-link-search={}/build", dst.display());
    println!("cargo:rustc-link-lib=flexfloat");
}
