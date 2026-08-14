{
  description = "MCP server and ingestion toolkit for a Postgres/pgvector code-intelligence database.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = lib.genAttrs systems;
      mkPackage = system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python314;
          pythonPackages = pkgs.python314Packages;
        in
        pythonPackages.buildPythonApplication {
          pname = "project-code-intelligence";
          version = "0.1.0";
          pyproject = true;

          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./LICENSE
              ./README.md
              ./pyproject.toml
              ./src
            ];
          };

          build-system = [
            pythonPackages.setuptools
          ];

          dependencies = [
            pythonPackages.psycopg
            pythonPackages.pydantic
            pythonPackages.rich
          ];

          nativeCheckInputs = [
            python
          ];

          pythonImportsCheck = [
            "project_code_intelligence"
            "project_code_intelligence.config"
            "project_code_intelligence.doctor"
            "project_code_intelligence.mcp"
            "project_code_intelligence.storage"
          ];

          meta = {
            description = "MCP server and ingestion toolkit for a Postgres/pgvector code-intelligence database.";
            homepage = "https://github.com/cvandesande/project-code-intelligence";
            license = lib.licenses.mit;
            mainProgram = "pci";
          };
        };
    in
    {
      packages = forAllSystems (system:
        let
          project-code-intelligence = mkPackage system;
        in
        {
          inherit project-code-intelligence;
          default = project-code-intelligence;
        });

      apps = forAllSystems (system:
        let
          package = self.packages.${system}.default;
          pciApp = {
            type = "app";
            program = "${package}/bin/pci";
            meta.description = "Run pci (use subcommands: index, doctor, mcp, ...).";
          };
        in
        {
          pci = pciApp;
          default = pciApp;
        });

      checks = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          package = self.packages.${system}.default;
        in
        {
          inherit package;
          doctor-smoke = pkgs.runCommand "project-code-intelligence-doctor-smoke"
            {
              nativeBuildInputs = [ package ];
            }
            ''
              export HOME="$TMPDIR/home"
              mkdir -p "$HOME"
              pci doctor --skip-db --embedding skip --json > "$out"
            '';
        });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python314.withPackages (ps: [
            ps.coverage
            ps.pip
            ps.psycopg
            ps.pydantic
            ps.rich
            ps.tomli
            ps.vulture
          ]);
          pip-audit = pkgs.writeShellApplication {
            name = "pip-audit";
            text = ''
              export PIPAPI_PYTHON_LOCATION="${python}/bin/python"
              exec ${pkgs.pip-audit}/bin/pip-audit "$@"
            '';
          };
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.basedpyright
              pkgs.bandit
              pip-audit
              pkgs.ruff
              pkgs.semgrep
              pkgs.shellcheck
              pkgs.shfmt
              pkgs.uv
            ];
            shellHook = ''
              export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        });
    };
}
