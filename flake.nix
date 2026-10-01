{
  description = "rustyrays dev environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    # Provides any rustc version, read from rust-toolchain.toml
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        overlays = [ rust-overlay.overlays.default ];
      };
      # Toolchain version, components and targets all come from rust-toolchain.toml
      rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
    in {
      # Entered by `nix develop`, or automatically by direnv via .envrc (`use flake`).
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          rust
          pkgs.cargo-flamegraph # profiling: `cargo flamegraph --bin raytrace -- params-small.json simple.json`
          pkgs.imagemagick # convert out/output.ppm to png
        ];
      };
    };
}
