# mlp(opencv-mlp)
To use opencv's mlp to implement classification is the aim of this project.
How to Build
1.Build OpenCV
2.Build Boost

	Build the 32-bit libraries
	This installs the Boost header files under C:\Boost\include\boost-(version), and the 32-bit libraries under C:\Boost\lib\i386. Note that the default location for the libraries is C:\Boost\lib but you’ll want to put them under an i386 directory if you plan to build for multiple architectures.

  1) Unzip Boost into a new directory.
  2) Start a 32-bit MSVC command prompt and change to the directory where Boost was unzipped.
  3) Run: bootstrap
  4) Run: b2 toolset=msvc-12.0 --build-type=complete --libdir=C:\Boost\lib\i386 install
	For Visual Studio 2012, use toolset=msvc-11.0
	For Visual Studio 2010, use toolset=msvc-10.0
  5) Add C:\Boost\include\boost-(version) to your include path.
  6) Add C:\Boost\lib\i386 to your libs path.

	Build the 64-bit libraries
	This installs the Boost header files under C:\Boost\include\boost-(version), and the 64-bit libraries under C:\Boost\lib\x64. Note that the default location for the libraries is C:\Boost\lib but you’ll want to put them under an x64 directory if you plan to build for multiple architectures.

  1) Unzip Boost into a new directory.
  2) Start a 64-bit MSVC command prompt and change to the directory where Boost was unzipped.
  3) Run: bootstrap
  4) Run: b2 toolset=msvc-12.0 --build-type=complete --libdir=C:\Boost\lib\x64 architecture=x86 address-model=64 install
	For Visual Studio 2012, use toolset=msvc-11.0
	For Visual Studio 2010, use toolset=msvc-10.0
  5) Add C:\Boost\include\boost-(version) to your include path.
  6) Add C:\Boost\lib\x64 to your libs path.

3.Set VC++ Directories
 

4.Link DLLs
 

5.PrepareData
$(mlp_root_dir)/TrainingData/$(prjname)/$(classA)
$(mlp_root_dir)/TrainingData/$(prjname)/$(classB)
$(mlp_root_dir)/TrainingData/$(prjname)/$(classC)
……
$(mlp_root_dir)/TrainiedData/$(prjname)
$(mlp_root_dir)/TestData/$(prjname)
$(mlp_root_dir)/OutData/$(prjname)/$(classA)
$(mlp_root_dir)/OutData/$(prjname)/$(classB)
……

I used boost1.62.0, opencv3.1.0 and VS2013 for this project.
Attention:
You can encounter odd errors when you connect x86 dlls to x64 project. Please be aware of it.

How to Use
1.Open CMD
2.Train
MLPTrain.exe nameofclassA nameofclassB
3.Test
MLPTest.exe nameofclassA nameofclassB testcount
