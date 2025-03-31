#!/bin/bash

VCPKG_DIR="$HOME/vcpkg"
BASHRC="$HOME/.bashrc"

# Clone vcpkg if it does not exist
if [ ! -d "$VCPKG_DIR" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$VCPKG_DIR"
    "$VCPKG_DIR/bootstrap-vcpkg.sh"
else
    echo "vcpkg already cloned. Skipping clone."
fi

# Check if VCPKG_ROOT is already in .bashrc
if grep -q "export VCPKG_ROOT=" "$BASHRC"; then
    echo "Updating existing VCPKG_ROOT in .bashrc"
    sed -i "s|export VCPKG_ROOT=.*|export VCPKG_ROOT=\"$VCPKG_DIR\"|" "$BASHRC"
else
    echo "Adding VCPKG_ROOT to .bashrc"
    echo " " >> "$BASHRC"
    echo "export VCPKG_ROOT=\"$VCPKG_DIR\"" >> "$BASHRC"
fi

# Reload .bashrc
echo "Done. Please run 'source ~/.bashrc' or open a new terminal."