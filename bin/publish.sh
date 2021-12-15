#!/usr/bin/env bash
cd "$(dirname "$0")" && cd ..
while true; do
  echo "Do you want to build the package?[Y/n]"
  read yn
  case ${yn} in
  [Yy] | "")
    bash bin/build.sh
    break
    ;;

  [Nn]) break ;;

  *) echo "Please answer y/n."
  esac
done

while true; do
  echo "Do you wish to publish the package?[Y/n]"
  read yn
  case ${yn} in
  [Yy] | "")
    while true; do
      echo "Do you wish to publish first on TestPyPI ('n' for PyPI)?[Y/n]"
      read yn
      case ${yn} in
      [Nn])
        twine upload dist/*
        exit
        ;;

      [Yy] | "")
        twine upload --repository testpypi dist/*
        while true; do
          echo "Do you wish to publish on PyPI?[Y/n]"
          read yn
          case ${yn} in
          [Yy] | "")
            twine upload dist/*
            exit
            ;;

          [Nn])
            exit
            ;;

          *) echo "Please answer y/n."
          esac
        done
        ;;

      *) echo "Please answer y/n."
      esac
    done
    ;;

  [Nn]) break ;;

  *) echo "Please answer y/n."
  esac
done
