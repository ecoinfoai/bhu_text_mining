{
  description = "formative-analysis — AI-powered formative assessment CLI toolkit";

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
          mecab
          jdk17_headless  # KoNLPy / JPype — JVM required
          tesseract       # pytesseract OCR backend
        ];

        env = {
          # Force uv to use Nix-provided Python (not download its own)
          UV_PYTHON_PREFERENCE = "only-system";
        };

        shellHook = ''
          # mecab: use Korean dictionary from mecab-ko-dic pip package
          DICDIR=$(python3 -c \
            "import mecab_ko_dic; print(mecab_ko_dic.DICDIR)" \
            2>/dev/null || true)
          if [ -n "$DICDIR" ] && [ -d "$DICDIR" ]; then
            export MECABRC=$(mktemp)
            echo "dicdir = $DICDIR" > "$MECABRC"
          fi

          # KoNLPy / JPype: explicit JAVA_HOME
          export JAVA_HOME="${pkgs.jdk17_headless}"
        '';

        # Shared libraries for native Python wheels
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib  # libstdc++
          pkgs.zbar              # pyzbar QR decoding
          pkgs.zlib              # general compression
          pkgs.mecab             # mecab-python3
        ];
      };
    };
}
