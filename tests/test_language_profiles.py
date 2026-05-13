from __future__ import annotations

import unittest

from project_code_intelligence.language_profiles import language_metadata_for_file, language_metadata_keys


class LanguageProfileTests(unittest.TestCase):
    def test_c_family_metadata_extracts_includes_macros_types_and_functions(self) -> None:
        metadata = language_metadata_for_file(
            "src/demo.c",
            "c",
            """
#include <stdio.h>
#include "demo.h"
#define DEMO_FLAG 1
struct demo_state { int value; };
static int demo_run(struct demo_state *state) {
    return state->value;
}
""",
        )

        self.assertEqual(metadata["c_family_system_includes"], ["stdio.h"])
        self.assertEqual(metadata["c_family_local_includes"], ["demo.h"])
        self.assertEqual(metadata["c_family_defines"], ["DEMO_FLAG"])
        self.assertEqual(metadata["c_family_types"], ["demo_state"])
        self.assertEqual(metadata["c_family_declared_functions"], ["demo_run"])

    def test_csharp_metadata_extracts_usings_namespaces_types_methods_and_attributes(self) -> None:
        metadata = language_metadata_for_file(
            "src/Demo.cs",
            "csharp",
            """
using System;
using static System.Math;

namespace Demo.App;

[ApiController]
public record Request(string Name);
public interface IRunner {}
public enum Mode { Fast }

public class Worker
{
    public async Task RunAsync() { await Task.CompletedTask; }
}
""",
        )

        self.assertEqual(metadata["csharp_usings"], ["System", "System.Math"])
        self.assertEqual(metadata["csharp_namespaces"], ["Demo.App"])
        self.assertEqual(metadata["csharp_classes"], ["Worker"])
        self.assertEqual(metadata["csharp_records"], ["Request"])
        self.assertEqual(metadata["csharp_interfaces"], ["IRunner"])
        self.assertEqual(metadata["csharp_enums"], ["Mode"])
        self.assertEqual(metadata["csharp_methods"], ["RunAsync"])
        self.assertEqual(metadata["csharp_attributes"], ["ApiController"])
        self.assertTrue(metadata["csharp_has_async"])

    def test_go_metadata_extracts_package_imports_symbols_and_receivers(self) -> None:
        metadata = language_metadata_for_file(
            "pkg/demo/demo.go",
            "go",
            """
package demo

import (
    "context"
    alias "net/http"
)

type Server struct{}
type Runner interface{}

func NewServer() *Server { return &Server{} }
func (s *Server) Serve(ctx context.Context) error { return nil }
""",
        )

        self.assertEqual(metadata["go_package"], "demo")
        self.assertEqual(metadata["go_imports"], ["context", "net/http"])
        self.assertEqual(metadata["go_functions"], ["NewServer"])
        self.assertEqual(metadata["go_methods"], ["Serve"])
        self.assertEqual(metadata["go_receiver_types"], ["Server"])
        self.assertEqual(metadata["go_structs"], ["Server"])
        self.assertEqual(metadata["go_interfaces"], ["Runner"])

    def test_javascript_typescript_metadata_extracts_imports_symbols_and_types(self) -> None:
        metadata = language_metadata_for_file(
            "src/app.ts",
            "typescript",
            """
import React from "react";
import type { Route } from "./router";
const fs = require("node:fs");

export interface Props { name: string }
export type Mode = "fast" | "safe";
export enum State { Ready }

export function render(props: Props) { return props.name; }
export const load = async () => fs.promises.readFile("demo.txt");
export class App {}
""",
        )

        self.assertEqual(metadata["js_imports"], ["react", "./router", "node:fs"])
        self.assertEqual(metadata["js_exports"], ["render", "load", "App"])
        self.assertEqual(metadata["js_functions"], ["render", "load"])
        self.assertEqual(metadata["js_classes"], ["App"])
        self.assertEqual(metadata["ts_interfaces"], ["Props"])
        self.assertEqual(metadata["ts_types"], ["Mode"])
        self.assertEqual(metadata["ts_enums"], ["State"])

    def test_java_metadata_extracts_package_imports_types_methods_and_annotations(self) -> None:
        metadata = language_metadata_for_file(
            "src/main/java/demo/App.java",
            "java",
            """
package demo.app;

import java.util.List;
import static java.util.Collections.emptyList;

@Component
public class App {
    public List<String> run() { return emptyList(); }
}

public interface Runner {}
public enum Mode { Fast }
public record Request(String name) {}
""",
        )

        self.assertEqual(metadata["jvm_package"], "demo.app")
        self.assertEqual(metadata["jvm_imports"], ["java.util.List", "java.util.Collections.emptyList"])
        self.assertEqual(metadata["java_classes"], ["App"])
        self.assertEqual(metadata["java_interfaces"], ["Runner"])
        self.assertEqual(metadata["java_enums"], ["Mode"])
        self.assertEqual(metadata["java_records"], ["Request"])
        self.assertEqual(metadata["java_methods"], ["run"])
        self.assertEqual(metadata["java_annotations"], ["Component"])

    def test_kotlin_metadata_extracts_package_imports_types_and_functions(self) -> None:
        metadata = language_metadata_for_file(
            "src/main/kotlin/demo/App.kt",
            "kotlin",
            """
package demo.app

import kotlinx.coroutines.delay

data class App(val name: String)
interface Runner
object Registry

suspend fun load() {
    delay(1)
}
""",
        )

        self.assertEqual(metadata["jvm_package"], "demo.app")
        self.assertEqual(metadata["jvm_imports"], ["kotlinx.coroutines.delay"])
        self.assertEqual(metadata["kotlin_classes"], ["App"])
        self.assertEqual(metadata["kotlin_interfaces"], ["Runner"])
        self.assertEqual(metadata["kotlin_objects"], ["Registry"])
        self.assertEqual(metadata["kotlin_functions"], ["load"])
        self.assertTrue(metadata["kotlin_has_suspend"])

    def test_lua_metadata_extracts_requires_modules_functions_and_locals(self) -> None:
        metadata = language_metadata_for_file(
            "luci/controller/demo.lua",
            "lua",
            """
module("luci.controller.demo", package.seeall)
local http = require "luci.http"

function index()
end

demo.run = function()
end
""",
        )

        self.assertEqual(metadata["lua_requires"], ["luci.http"])
        self.assertEqual(metadata["lua_modules"], ["luci.controller.demo"])
        self.assertEqual(metadata["lua_functions"], ["index", "demo.run"])
        self.assertEqual(metadata["lua_locals"], ["http"])

    def test_perl_metadata_extracts_packages_modules_requires_and_subroutines(self) -> None:
        metadata = language_metadata_for_file(
            "scripts/checkpatch.pl",
            "perl",
            """
use strict;
use warnings;
use Getopt::Long;
require "helpers.pl";

package OpenWrt::Check;

sub run_check {
    return 1;
}
""",
        )

        self.assertEqual(metadata["perl_packages"], ["OpenWrt::Check"])
        self.assertEqual(metadata["perl_modules"], ["strict", "warnings", "Getopt::Long"])
        self.assertEqual(metadata["perl_requires"], ["helpers.pl"])
        self.assertEqual(metadata["perl_subroutines"], ["run_check"])
        self.assertTrue(metadata["perl_uses_strict"])
        self.assertTrue(metadata["perl_uses_warnings"])

    def test_openwrt_format_metadata_extracts_build_dsl_facts(self) -> None:
        autotools = language_metadata_for_file(
            "src/aclocal.m4",
            "autotools",
            """
AC_DEFUN([OPENWRT_CHECK], [AC_REQUIRE([AC_PROG_CC])])
AM_INIT_AUTOMAKE
""",
        )
        linker = language_metadata_for_file(
            "image/linker.lds",
            "linker_script",
            """
ENTRY(_start)
SECTIONS {
  .text : { *(.text*) }
  PROVIDE(end = .);
}
""",
        )
        boot = language_metadata_for_file(
            "image/board.bootscript",
            "boot_script",
            """
bootcmd=run boot_openwrt
setenv bootargs console=ttyS0
bootm ${kernel_addr}
""",
        )
        awk = language_metadata_for_file("include/scan.awk", "awk", "function scan_file(path) { print path }")
        lex = language_metadata_for_file("scripts/config/lexer.l", "lex", "%x STRING COMMENT\n%%\n")
        yacc = language_metadata_for_file(
            "scripts/config/parser.y", "yacc", "%token T_WORD T_EOL\n%%\nmenu: T_WORD ;\n"
        )

        self.assertEqual(autotools["autotools_macros"], ["AC_DEFUN", "AC_REQUIRE", "AC_PROG_CC", "AM_INIT_AUTOMAKE"])
        self.assertEqual(autotools["autotools_definitions"], ["OPENWRT_CHECK"])
        self.assertEqual(linker["linker_entry_symbols"], ["_start"])
        self.assertEqual(linker["linker_sections"], ["text"])
        self.assertEqual(linker["linker_provided_symbols"], ["end"])
        self.assertEqual(boot["boot_script_variables"], ["bootcmd"])
        self.assertEqual(boot["boot_script_commands"], ["bootcmd", "setenv", "bootm"])
        self.assertEqual(awk["awk_functions"], ["scan_file"])
        self.assertEqual(lex["lex_start_conditions"], ["STRING", "COMMENT"])
        self.assertEqual(yacc["yacc_tokens"], ["T_WORD", "T_EOL"])
        self.assertEqual(yacc["yacc_rules"], ["menu"])

    def test_php_metadata_extracts_namespaces_types_functions_and_attributes(self) -> None:
        metadata = language_metadata_for_file(
            "src/Demo.php",
            "php",
            """
<?php
namespace Demo\\App;

use Psr\\Log\\LoggerInterface;
use function Demo\\helper;

#[Route]
final class Controller {}
interface Runner {}
trait Logs {}
enum Mode { case Fast; }

function boot(): void {}
""",
        )

        self.assertEqual(metadata["php_namespaces"], ["Demo\\App"])
        self.assertEqual(metadata["php_uses"], ["Psr\\Log\\LoggerInterface", "Demo\\helper"])
        self.assertEqual(metadata["php_classes"], ["Controller"])
        self.assertEqual(metadata["php_interfaces"], ["Runner"])
        self.assertEqual(metadata["php_traits"], ["Logs"])
        self.assertEqual(metadata["php_enums"], ["Mode"])
        self.assertEqual(metadata["php_functions"], ["boot"])
        self.assertEqual(metadata["php_attributes"], ["Route"])

    def test_ruby_metadata_extracts_requires_modules_classes_and_methods(self) -> None:
        metadata = language_metadata_for_file(
            "lib/demo.rb",
            "ruby",
            """
require "json"
require_relative "support"

module Demo
  class Worker
    def self.build
      new
    end

    def run!
    end
  end
end
""",
        )

        self.assertEqual(metadata["ruby_requires"], ["json", "support"])
        self.assertEqual(metadata["ruby_modules"], ["Demo"])
        self.assertEqual(metadata["ruby_classes"], ["Worker"])
        self.assertEqual(metadata["ruby_methods"], ["run!"])
        self.assertEqual(metadata["ruby_singleton_methods"], ["build"])

    def test_document_metadata_extracts_markdown_and_rst_structure(self) -> None:
        markdown = language_metadata_for_file(
            "docs/demo.md",
            "doc",
            """
# Demo
See [API](api.md).

```python
print("demo")
```
""",
        )
        rst = language_metadata_for_file(
            "docs/demo.rst",
            "doc",
            """
Demo Guide
==========

See `Docs <https://example.invalid>`_.

.. code-block:: shell
""",
        )

        self.assertEqual(markdown["doc_headings"], ["Demo"])
        self.assertEqual(markdown["doc_links"], ["api.md"])
        self.assertEqual(markdown["doc_fenced_languages"], ["python"])
        self.assertEqual(rst["doc_headings"], ["Demo Guide"])
        self.assertEqual(rst["doc_links"], ["https://example.invalid"])
        self.assertEqual(rst["doc_fenced_languages"], ["shell"])

    def test_xml_and_sql_metadata_extracts_structural_facts(self) -> None:
        xml = language_metadata_for_file(
            "config/app.xml",
            "xml",
            """
<?xml version="1.0"?>
<service name="demo">
  <endpoint path="/api" />
</service>
""",
        )
        sql = language_metadata_for_file(
            "db/schema.sql",
            "sql",
            """
CREATE TABLE users (id integer primary key);
CREATE INDEX users_id_idx ON users (id);
INSERT INTO users VALUES (1);
CREATE FUNCTION refresh_users() RETURNS void AS $$ SELECT 1; $$ LANGUAGE sql;
""",
        )

        self.assertEqual(xml["xml_root"], "service")
        self.assertEqual(xml["xml_elements"], ["service", "endpoint"])
        self.assertEqual(xml["xml_attributes"], ["version", "name", "path"])
        self.assertEqual(sql["sql_operations"], ["create", "insert", "select"])
        self.assertEqual(sql["sql_tables"], ["users"])
        self.assertEqual(sql["sql_indexes"], ["users_id_idx"])
        self.assertEqual(sql["sql_functions"], ["refresh_users"])

    def test_infra_metadata_extracts_docker_terraform_and_packer_facts(self) -> None:
        dockerfile = language_metadata_for_file(
            "Dockerfile",
            "dockerfile",
            """
# comment before the first instruction
FROM python:3.13-slim AS runtime
COPY src /app/src
EXPOSE 8080/tcp
CMD ["python", "-m", "demo"]
""",
        )
        terraform = language_metadata_for_file(
            "infra/main.tf",
            "terraform",
            """
provider "aws" {}
resource "aws_s3_bucket" "logs" {}
data "aws_iam_policy_document" "read" {}
module "network" {}
variable "region" {}
output "bucket" {}
""",
        )
        packer = language_metadata_for_file(
            "images/debian.pkr.hcl",
            "packer",
            """
packer {}
source "amazon-ebs" "debian" {}
build {}
variable "region" {}
""",
        )

        self.assertEqual(dockerfile["docker_base_images"], ["python:3.13-slim"])
        self.assertEqual(dockerfile["docker_stages"], ["runtime"])
        self.assertEqual(dockerfile["docker_instructions"], ["FROM", "COPY", "EXPOSE", "CMD"])
        self.assertEqual(dockerfile["docker_exposed_ports"], ["8080/tcp"])
        self.assertEqual(dockerfile["docker_copy_sources"], ["src"])
        self.assertEqual(terraform["terraform_resources"], ["aws_s3_bucket.logs"])
        self.assertEqual(terraform["terraform_data_sources"], ["aws_iam_policy_document.read"])
        self.assertEqual(terraform["terraform_modules"], ["network"])
        self.assertEqual(terraform["terraform_variables"], ["region"])
        self.assertEqual(terraform["terraform_outputs"], ["bucket"])
        self.assertEqual(terraform["terraform_providers"], ["aws"])
        self.assertEqual(packer["packer_sources"], ["amazon-ebs.debian"])
        self.assertEqual(packer["packer_blocks"], ["packer", "build"])
        self.assertEqual(packer["packer_variables"], ["region"])

    def test_protobuf_and_objective_c_metadata_extracts_interfaces(self) -> None:
        protobuf = language_metadata_for_file(
            "api/demo.proto",
            "protobuf",
            """
syntax = "proto3";
package demo.v1;
import "google/protobuf/timestamp.proto";
message Request {}
enum Mode { MODE_UNSPECIFIED = 0; }
service DemoService {
  rpc Run(Request) returns (Request);
}
""",
        )
        objc = language_metadata_for_file(
            "src/AppDelegate.m",
            "objective_c",
            """
#import <Foundation/Foundation.h>
@protocol DemoDelegate
@end
@interface AppDelegate : NSObject
- (void)applicationDidFinishLaunching:(id)sender;
@end
@implementation AppDelegate
@end
""",
        )

        self.assertEqual(protobuf["proto_package"], "demo.v1")
        self.assertEqual(protobuf["proto_imports"], ["google/protobuf/timestamp.proto"])
        self.assertEqual(protobuf["proto_messages"], ["Request"])
        self.assertEqual(protobuf["proto_services"], ["DemoService"])
        self.assertEqual(protobuf["proto_rpcs"], ["Run"])
        self.assertEqual(protobuf["proto_enums"], ["Mode"])
        self.assertEqual(objc["objc_imports"], ["Foundation/Foundation.h"])
        self.assertEqual(objc["objc_interfaces"], ["AppDelegate"])
        self.assertEqual(objc["objc_implementations"], ["AppDelegate"])
        self.assertEqual(objc["objc_protocols"], ["DemoDelegate"])
        self.assertEqual(objc["objc_methods"], ["applicationDidFinishLaunching:"])

    def test_build_system_metadata_extracts_cmake_and_meson_facts(self) -> None:
        cmake = language_metadata_for_file(
            "CMakeLists.txt",
            "cmake",
            """
cmake_minimum_required(VERSION 3.25)
project(Demo)
find_package(OpenSSL REQUIRED)
add_library(demo src/demo.c)
add_executable(demo-cli src/main.c)
""",
        )
        meson = language_metadata_for_file(
            "meson.build",
            "meson",
            """
project('demo', 'c')
openssl = dependency('openssl')
library('demo', 'src/demo.c')
executable('demo-cli', 'src/main.c')
""",
        )

        self.assertEqual(
            cmake["cmake_commands"],
            ["cmake_minimum_required", "project", "find_package", "add_library", "add_executable"],
        )
        self.assertEqual(cmake["cmake_projects"], ["Demo"])
        self.assertEqual(cmake["cmake_packages"], ["OpenSSL"])
        self.assertEqual(cmake["cmake_targets"], ["demo", "demo-cli"])
        self.assertEqual(meson["meson_projects"], ["demo"])
        self.assertEqual(meson["meson_dependencies"], ["openssl"])
        self.assertEqual(meson["meson_targets"], ["demo", "demo-cli"])


class AdditionalLanguageProfileTests(unittest.TestCase):
    def test_swift_metadata_extracts_imports_types_functions_and_extensions(self) -> None:
        metadata = language_metadata_for_file(
            "Sources/Demo/App.swift",
            "swift",
            """
import Foundation
import SwiftUI

public protocol Runner {}
struct Request { let name: String }
enum Mode { case fast }
actor Worker {}

extension Worker: Runner {}

func load() async throws {}
private var cache = [String: Request]()
""",
        )

        self.assertEqual(metadata["swift_imports"], ["Foundation", "SwiftUI"])
        self.assertEqual(metadata["swift_structs"], ["Request"])
        self.assertEqual(metadata["swift_enums"], ["Mode"])
        self.assertEqual(metadata["swift_protocols"], ["Runner"])
        self.assertEqual(metadata["swift_actors"], ["Worker"])
        self.assertEqual(metadata["swift_extensions"], ["Worker"])
        self.assertEqual(metadata["swift_functions"], ["load"])
        self.assertEqual(metadata["swift_properties"], ["cache"])
        self.assertTrue(metadata["swift_has_async"])

    def test_web_metadata_extracts_html_css_and_component_facts(self) -> None:
        html = language_metadata_for_file(
            "web/index.html",
            "html",
            """
<!doctype html>
<html>
  <body>
    <my-widget data-id="1" src="/ignored"></my-widget>
    <script type="module" src="/assets/app.js"></script>
  </body>
</html>
""",
        )
        css = language_metadata_for_file(
            "web/app.scss",
            "scss",
            """
@import "theme.css";
:root { --accent: red; }
.app, #main { color: var(--accent); }
@mixin card { border: 1px solid currentColor; }
.panel { @include card; }
""",
        )
        component = language_metadata_for_file(
            "web/App.vue",
            "vue",
            """
<template><Button class="primary" /></template>
<script setup lang="ts">
import Button from "./Button.vue";
</script>
<style scoped lang="scss">
button.primary { color: red; }
</style>
""",
        )

        self.assertEqual(html["html_root"], "html")
        self.assertEqual(html["html_custom_elements"], ["my-widget"])
        self.assertEqual(html["html_attributes"], ["data-id", "src", "type"])
        self.assertEqual(html["html_links"], ["/ignored", "/assets/app.js"])
        self.assertEqual(css["css_imports"], ["theme.css"])
        self.assertEqual(css["css_variables"], ["--accent"])
        self.assertEqual(css["css_selectors"], [":root", ".app", "#main", ".panel"])
        self.assertEqual(css["scss_mixins"], ["card"])
        self.assertEqual(component["component_blocks"], ["template", "script", "style"])
        self.assertEqual(component["component_script_languages"], ["ts"])
        self.assertEqual(component["component_style_languages"], ["scss"])
        self.assertEqual(component["component_imports"], ["./Button.vue"])

    def test_graphql_metadata_extracts_schema_and_operation_facts(self) -> None:
        metadata = language_metadata_for_file(
            "api/schema.graphql",
            "graphql",
            """
scalar DateTime
interface Node { id: ID! }
type User implements Node { id: ID! }
input UserInput { name: String! }
enum Role { ADMIN USER }
union SearchResult = User

query GetUser { user { id } }
fragment UserFields on User { id }
""",
        )

        self.assertEqual(metadata["graphql_operations"], ["GetUser"])
        self.assertEqual(metadata["graphql_operation_kinds"], ["query"])
        self.assertEqual(metadata["graphql_fragments"], ["UserFields"])
        self.assertEqual(metadata["graphql_fragment_types"], ["User"])
        self.assertEqual(metadata["graphql_types"], ["User"])
        self.assertEqual(metadata["graphql_inputs"], ["UserInput"])
        self.assertEqual(metadata["graphql_interfaces"], ["Node"])
        self.assertEqual(metadata["graphql_enums"], ["Role"])
        self.assertEqual(metadata["graphql_unions"], ["SearchResult"])
        self.assertEqual(metadata["graphql_scalars"], ["DateTime"])

    def test_bazel_groovy_and_powershell_metadata_extracts_build_and_script_facts(self) -> None:
        bazel = language_metadata_for_file(
            "BUILD.bazel",
            "bazel",
            """
load("//tools:defs.bzl", "custom_rule", "other_rule")
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "core",
)
def helper(name):
    pass
""",
        )
        groovy = language_metadata_for_file(
            "build.gradle",
            "groovy",
            """
plugins {
    id 'java'
}
import groovy.json.JsonSlurper
class BuildLogic {}
implementation 'org.example:lib:1.0'
tasks.register("checkDemo")
""",
        )
        powershell = language_metadata_for_file(
            "scripts/deploy.ps1",
            "powershell",
            """
Import-Module Pester
function Invoke-Deploy {
    [Parameter(Mandatory=$true)]
    [string]$Target
    Get-ChildItem $Target
}
""",
        )

        self.assertEqual(bazel["starlark_loads"], ["//tools:defs.bzl"])
        self.assertEqual(bazel["starlark_loaded_symbols"], ["custom_rule", "other_rule"])
        self.assertEqual(bazel["starlark_functions"], ["helper"])
        self.assertEqual(bazel["bazel_rules"], ["cc_library"])
        self.assertEqual(bazel["bazel_targets"], ["core"])
        self.assertTrue(bazel["bazel_has_package_declaration"])
        self.assertEqual(groovy["groovy_imports"], ["groovy.json.JsonSlurper"])
        self.assertEqual(groovy["groovy_classes"], ["BuildLogic"])
        self.assertEqual(groovy["gradle_plugins"], ["java"])
        self.assertEqual(groovy["gradle_dependencies"], ["org.example:lib:1.0"])
        self.assertEqual(groovy["gradle_tasks"], ["checkDemo"])
        self.assertEqual(powershell["powershell_functions"], ["Invoke-Deploy"])
        self.assertEqual(powershell["powershell_parameters"], ["Target"])
        self.assertEqual(powershell["powershell_imported_modules"], ["Pester"])
        self.assertEqual(powershell["powershell_cmdlets"], ["Import-Module", "Get-ChildItem"])

    def test_scala_beam_and_zig_metadata_extracts_portable_facts(self) -> None:
        scala = language_metadata_for_file(
            "src/main/scala/demo/App.scala",
            "scala",
            """
package demo.app
import scala.concurrent.Future
trait Runner
case class Request(name: String)
object App {
  val name = "demo"
  def run(): Unit = ()
}
""",
        )
        elixir = language_metadata_for_file(
            "lib/demo/worker.ex",
            "elixir",
            """
defmodule Demo.Worker do
  use GenServer
  alias Demo.Context
  import Enum
  require Logger
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts)
  defp normalize(value), do: value
  defmacro loggable(expr), do: expr
end
""",
        )
        erlang = language_metadata_for_file(
            "src/demo_worker.erl",
            "erlang",
            """
-module(demo_worker).
-behaviour(gen_server).
-export([start/0, stop/1]).
-record(state, {value}).
start() -> ok.
stop(_Pid) -> ok.
""",
        )
        zig = language_metadata_for_file(
            "src/main.zig",
            "zig",
            """
const std = @import("std");
pub const App = struct {};
const Mode = enum { fast };
pub fn main() !void {}
test "loads config" {}
""",
        )

        self.assertEqual(scala["scala_package"], "demo.app")
        self.assertEqual(scala["scala_imports"], ["scala.concurrent.Future"])
        self.assertEqual(scala["scala_classes"], ["Request"])
        self.assertEqual(scala["scala_objects"], ["App"])
        self.assertEqual(scala["scala_traits"], ["Runner"])
        self.assertEqual(scala["scala_defs"], ["run"])
        self.assertEqual(scala["scala_values"], ["name"])
        self.assertEqual(elixir["elixir_modules"], ["Demo.Worker"])
        self.assertEqual(elixir["elixir_aliases"], ["Demo.Context"])
        self.assertEqual(elixir["elixir_imports"], ["Enum"])
        self.assertEqual(elixir["elixir_requires"], ["Logger"])
        self.assertEqual(elixir["elixir_uses"], ["GenServer"])
        self.assertEqual(elixir["elixir_functions"], ["start_link"])
        self.assertEqual(elixir["elixir_private_functions"], ["normalize"])
        self.assertEqual(elixir["elixir_macros"], ["loggable"])
        self.assertEqual(erlang["erlang_module"], "demo_worker")
        self.assertEqual(erlang["erlang_exports"], ["start/0", "stop/1"])
        self.assertEqual(erlang["erlang_records"], ["state"])
        self.assertEqual(erlang["erlang_behaviours"], ["gen_server"])
        self.assertEqual(erlang["erlang_functions"], ["start", "stop"])
        self.assertEqual(zig["zig_imports"], ["std"])
        self.assertEqual(zig["zig_functions"], ["main"])
        self.assertEqual(zig["zig_structs"], ["App"])
        self.assertEqual(zig["zig_enums"], ["Mode"])
        self.assertEqual(zig["zig_tests"], ["loads config"])

    def test_python_metadata_uses_ast_when_available(self) -> None:
        metadata = language_metadata_for_file(
            "src/project/demo.py",
            "python",
            """
import asyncio
from pathlib import Path

@decorator
class Worker:
    async def run(self) -> None:
        await asyncio.sleep(0)
""",
        )

        self.assertEqual(metadata["python_module"], "project.demo")
        self.assertEqual(metadata["python_imports"], ["asyncio", "pathlib"])
        self.assertEqual(metadata["python_classes"], ["Worker"])
        self.assertEqual(metadata["python_functions"], ["run"])
        self.assertEqual(metadata["python_decorators"], ["decorator"])
        self.assertTrue(metadata["python_has_async"])

    def test_rust_metadata_extracts_items_and_unsafe_marker(self) -> None:
        metadata = language_metadata_for_file(
            "src/lib.rs",
            "rust",
            """
pub mod io;
use std::sync::Arc;

pub struct Engine;
pub enum Mode { Fast }
pub trait Run { fn run(&self); }
impl Engine { pub fn new() -> Self { Self } }
unsafe fn touch_raw(ptr: *const u8) -> u8 { *ptr }
""",
        )

        self.assertEqual(metadata["rust_modules"], ["io"])
        self.assertEqual(metadata["rust_uses"], ["std::sync::Arc"])
        self.assertEqual(metadata["rust_functions"], ["touch_raw"])
        self.assertEqual(metadata["rust_structs"], ["Engine"])
        self.assertEqual(metadata["rust_enums"], ["Mode"])
        self.assertEqual(metadata["rust_traits"], ["Run"])
        self.assertEqual(metadata["rust_impls"], ["Engine"])
        self.assertTrue(metadata["rust_uses_unsafe"])

    def test_shell_metadata_extracts_functions_sources_exports_and_commands(self) -> None:
        metadata = language_metadata_for_file(
            "files/etc/init.d/demo",
            "shell",
            """#!/bin/sh /etc/rc.common
. /lib/functions.sh
export DEMO_FLAG=1

start_service() {
    procd_open_instance
    uci get demo.main.enabled
}

stop() {
    killall demo
}
""",
        )

        self.assertEqual(metadata["shell_shebang"], "#!/bin/sh /etc/rc.common")
        self.assertEqual(metadata["shell_functions"], ["start_service", "stop"])
        self.assertEqual(metadata["shell_service_functions"], ["start_service", "stop"])
        self.assertEqual(metadata["shell_sources"], ["/lib/functions.sh"])
        self.assertEqual(metadata["shell_exports"], ["DEMO_FLAG"])
        self.assertEqual(metadata["shell_commands"], ["procd_open_instance", "uci", "killall"])

    def test_language_metadata_keys_are_registered_for_embedding_metadata(self) -> None:
        keys = language_metadata_keys()

        self.assertIn("c_family_local_includes", keys)
        self.assertIn("csharp_classes", keys)
        self.assertIn("go_package", keys)
        self.assertIn("java_classes", keys)
        self.assertIn("js_imports", keys)
        self.assertIn("lua_functions", keys)
        self.assertIn("perl_subroutines", keys)
        self.assertIn("php_classes", keys)
        self.assertIn("ruby_classes", keys)
        self.assertIn("doc_headings", keys)
        self.assertIn("sql_tables", keys)
        self.assertIn("xml_elements", keys)
        self.assertIn("docker_base_images", keys)
        self.assertIn("terraform_resources", keys)
        self.assertIn("packer_sources", keys)
        self.assertIn("proto_messages", keys)
        self.assertIn("objc_interfaces", keys)
        self.assertIn("cmake_targets", keys)
        self.assertIn("meson_targets", keys)
        self.assertIn("swift_structs", keys)
        self.assertIn("html_elements", keys)
        self.assertIn("css_selectors", keys)
        self.assertIn("component_blocks", keys)
        self.assertIn("graphql_types", keys)
        self.assertIn("bazel_targets", keys)
        self.assertIn("groovy_classes", keys)
        self.assertIn("powershell_functions", keys)
        self.assertIn("scala_classes", keys)
        self.assertIn("elixir_modules", keys)
        self.assertIn("erlang_exports", keys)
        self.assertIn("zig_functions", keys)
        self.assertIn("linker_sections", keys)
        self.assertIn("python_module", keys)
        self.assertIn("rust_functions", keys)
        self.assertIn("shell_functions", keys)


if __name__ == "__main__":
    _ = unittest.main()
