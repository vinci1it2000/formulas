#!/usr/bin/env bash
cd "$(dirname "$0")"
rm -vrf ./build/* ./dist/* ./*.pyc ./*.tgz ./*.egg-info
export ENABLE_SETUP_LONG_DESCRIPTION="TRUE"
python setup.py sdist bdist_wheel -v

while true; do
    echo "Do you wish to publish the package?[Y/n]"
    read yn
    case ${yn} in
        [Yy]|"")
        while true; do
            echo "Do you wish to publish on PyPI ('n' for TestPyPI)?[Y/n]"
            read yn
            case ${yn} in
                [Yy]|"")
                twine upload dist/*
                exit;;

                [Nn])
                twine upload --repository testpypi dist/*
                exit;;

                *) echo "Please answer y/n.";;
            esac
        done;;

        [Nn]) break;;

        *) echo "Please answer y/n.";;
    esac
done
