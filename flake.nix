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
          python311Packages.pyqt5
          uv
          bun
          mecab
          jdk  # KoNLPy (network_analysis.py) 의존성 — JPype가 JVM을 필요로 함
        ];
        # python312 removed from local shell — CI matrix covers 3.12 testing.

        # Force uv to use Nix-provided Python (not download its own)
        env = {
          UV_PYTHON_PREFERENCE = "only-system";
        };

        # mecab: use Korean dictionary from mecab-ko-dic pip package
        shellHook = ''
          DICDIR=$(python3 -c "import mecab_ko_dic; print(mecab_ko_dic.DICDIR)" 2>/dev/null || true)
          if [ -n "$DICDIR" ] && [ -d "$DICDIR" ]; then
            export MECABRC=$(mktemp)
            echo "dicdir = $DICDIR" > "$MECABRC"
          fi

          # KoNLPy / JPype: JAVA_HOME 명시 설정
          export JAVA_HOME="${pkgs.jdk}"
        '';

        # Provide libstdc++ and Qt5/zlib for PyQt5 via pip
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.qt5.qtbase
          pkgs.zbar
          pkgs.zlib
          pkgs.mecab
        ];
      };
    };
}
