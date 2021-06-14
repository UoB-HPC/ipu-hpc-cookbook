#Targeting the `IPUModel` emulator

`IPUModel` devices allow you to develop and run your IPU code on 
a CPU. This is great for getting started without a real IPU, or 
for writing test code.

To target the `IPUModel` instead of a real IPU, change your device selection code to
something like

```C++
poplar::Device getIpuModel(int numIpus, int tilesPerIpu) {
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIpus;
    ipuModel.tilesPerIPU = tilesPerIpu;
    return ipuModel.createDevice();
}
```

Some things to note:
* You can create 'small' virtual IPU devices with a limited number of tiles,
  which are useful when starting to scale an algorithm (get it running on 2, then 4, then something odd like 13 tiles before you try for thousands).
* We've noted some differences between `IPUModel` and IPU runs. For example, it's possible for your code to run on the model and crash on the IPU. Always perform your final testing on real devices.
* The executables for real IPUs are not runnable on IPUModels (`popc` targets a different instruction set during compilation)
* `IPUModel` code will run a lot slower than IPU code.
* If you want accurate IPU timings reflected in profiles generated from `IPUModel` code, you'll need to accurately
  set the number of cycles a vertex is expected to take.
* You can still capture profile output and inspect things in the Poplar GraphVision tools. This is really useful for understanding
the cost of communication between a small number of IPU tiles. 