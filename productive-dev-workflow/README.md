# A productive development workflow
Indulge us here for a small preach about taking the time to set yourself up for success with
IPU development. If you're SSHing into a remote server, typing your code in `vi` without code
inspection and autocomplete tools, manually building your application, and `tar`ring up artifacts
and copying them over to your IPU host, you'll have an unnecessarily frustrating time. You shouldn't
have to suffer in this way.

A productive development workflow has minimal feedback cycles. This means that you can take small,
demonstrably correct steps in the construction of your program, with minimal friction in building
code, deploying it to a device, setting breakpoints and watches in debug mode, and writing and running
tests. In other words, we really encourage the use of an IDE such as CLion or VSCode to make this simpler. 

We also really encourage the use of version control systems such as git during development, along
with a disciplined approach to using it, and the use of a modern make system that allows for
reproducible builds.

It's unlikely that you'll be developing on the same system that hosts your target
IPUs, which implies that you need to have setup remote deployment to your target IPU.

Here are some resources on setting up CLion for remote deployment:
* https://blog.jetbrains.com/clion/2019/03/webinar-recording-remote-development-with-clion/?gclid=CjwKCAjwwqaGBhBKEiwAMk-FtOlJpVVNtKfDd3RWQQILXE8BcJmP_0Oo5PsSCjdGhso9hreZj5udaRoCXVwQAvD_BwE&gclsrc=aw.ds
* https://www.jetbrains.com/help/clion/remote-projects-support.html

In CLion, remember to:
* create remote toolchain with your IPU server credentials 
* add the (remote) Graphcore SDK include path to the known header path for 
IDE autocomplete,
* the `PATH_TO_POPLAR_SDK/poplar-..../lib` path to your run configuration's
`LD_LIBRARY_PATH` environment variable
* Set up your remote CMake so that the CMAKE_INCLUDE_PATH includes the path to the poplar header,
 the CMAKE_PREFIX_PATH includes the path to the poplar libraries as show in the image below


ours is

```sh
POPLAR_ROOT=/home/thorbenl/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64
CMAKE_PREFIX_PATH=/home/your_username/poplar_sdk/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64:/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64;
LIBRARY_PATH=/home/your_username/poplar_sdk/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/lib:/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/lib;
LD_LIBRARY_PATH=/home/your_username/poplar_sdk/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/lib:/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/lib;
POPLAR_SDK_ENABLED=/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64;
CPATH=/home/your_username/poplar_sdk/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/include:/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/include;
OMPI_CPPFLAGS=-I/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/include/openmpi;
PYTHONPATH=/home/your_username/poplar_sdk/popart-ubuntu_18_04-2.0.0+108156-165bbd8a64/python:/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/python:/home/your_username/poplar_sdk/poplar-ubuntu_18_04-2.0.0+108156-165bbd8a64/lib/python
```

(replace `your_username` and the path to the poplar SDK you have installed)

* Also remember to modify the LD_LIBRARY_PATH env var for your runnable target
  configuratiin to include the path to the Graphcore libs

![An example of setting up remote cmake paths][remote-cmake-paths]

[remote-cmake-paths]: ./remote-cmake-paths.png "Setting up remote cmake paths in CLion"



