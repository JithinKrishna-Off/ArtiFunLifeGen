@echo off
cd /d "%~dp0"
cargo build --release -p alife-app
