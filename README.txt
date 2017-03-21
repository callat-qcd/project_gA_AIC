Compiling json1 Loadable extension
http://sqlite.org/loadext.html#build

YourCode.c = json1.c

Loadable extensions are C-code. To compile them on most unix-like operating systems, the usual command is something like this:
gcc -g -fPIC -shared YourCode.c -o YourCode.so

Macs are unix-like, but they do not follow the usual shared library conventions. To compile a shared library on a Mac, use a command like this:
gcc -g -fPIC -dynamiclib YourCode.c -o YourCode.dylib
