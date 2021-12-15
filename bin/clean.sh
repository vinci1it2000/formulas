#!/bin/sh
cd "$(dirname "$0")" && cd ..
rm -vrf ./py ./mpy ./dist ./build
rm -vrf ./*.pyc ./*.tgz ./*.egg-info ./MANIFEST ./*.py.exp ./*.py.out
