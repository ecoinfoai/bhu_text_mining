# ~/localgit/project-developers/flake.nix
{
  description = "Project development tools";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python311
          uv
          bun
        ];
        # python312 removed from local shell — CI matrix covers 3.12 testing.

        # Force uv to use Nix-provided Python (not download its own)
        env = {
          UV_PYTHON_PREFERENCE = "only-system";
        };

        # Provide libstdc++ for native node modules (Backlog.md)
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
        ];
      };
    };
}
