# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2019.1.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2019.1.2\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\pi\Desktop\Aurilius

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\pi\Desktop\Aurilius\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/pin.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pin.dir/flags.make

CMakeFiles/pin.dir/src/main.obj: CMakeFiles/pin.dir/flags.make
CMakeFiles/pin.dir/src/main.obj: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\pi\Desktop\Aurilius\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pin.dir/src/main.obj"
	C:\TDM-GCC-64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\pin.dir\src\main.obj -c C:\Users\pi\Desktop\Aurilius\src\main.cpp

CMakeFiles/pin.dir/src/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pin.dir/src/main.i"
	C:\TDM-GCC-64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\pi\Desktop\Aurilius\src\main.cpp > CMakeFiles\pin.dir\src\main.i

CMakeFiles/pin.dir/src/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pin.dir/src/main.s"
	C:\TDM-GCC-64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\pi\Desktop\Aurilius\src\main.cpp -o CMakeFiles\pin.dir\src\main.s

CMakeFiles/pin.dir/src/Vector/Vector2d.obj: CMakeFiles/pin.dir/flags.make
CMakeFiles/pin.dir/src/Vector/Vector2d.obj: ../src/Vector/Vector2d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\pi\Desktop\Aurilius\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pin.dir/src/Vector/Vector2d.obj"
	C:\TDM-GCC-64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\pin.dir\src\Vector\Vector2d.obj -c C:\Users\pi\Desktop\Aurilius\src\Vector\Vector2d.cpp

CMakeFiles/pin.dir/src/Vector/Vector2d.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pin.dir/src/Vector/Vector2d.i"
	C:\TDM-GCC-64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\pi\Desktop\Aurilius\src\Vector\Vector2d.cpp > CMakeFiles\pin.dir\src\Vector\Vector2d.i

CMakeFiles/pin.dir/src/Vector/Vector2d.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pin.dir/src/Vector/Vector2d.s"
	C:\TDM-GCC-64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\pi\Desktop\Aurilius\src\Vector\Vector2d.cpp -o CMakeFiles\pin.dir\src\Vector\Vector2d.s

CMakeFiles/pin.dir/src/Vector/Vector3d.obj: CMakeFiles/pin.dir/flags.make
CMakeFiles/pin.dir/src/Vector/Vector3d.obj: ../src/Vector/Vector3d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\pi\Desktop\Aurilius\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/pin.dir/src/Vector/Vector3d.obj"
	C:\TDM-GCC-64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\pin.dir\src\Vector\Vector3d.obj -c C:\Users\pi\Desktop\Aurilius\src\Vector\Vector3d.cpp

CMakeFiles/pin.dir/src/Vector/Vector3d.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pin.dir/src/Vector/Vector3d.i"
	C:\TDM-GCC-64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\pi\Desktop\Aurilius\src\Vector\Vector3d.cpp > CMakeFiles\pin.dir\src\Vector\Vector3d.i

CMakeFiles/pin.dir/src/Vector/Vector3d.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pin.dir/src/Vector/Vector3d.s"
	C:\TDM-GCC-64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\pi\Desktop\Aurilius\src\Vector\Vector3d.cpp -o CMakeFiles\pin.dir\src\Vector\Vector3d.s

# Object files for target pin
pin_OBJECTS = \
"CMakeFiles/pin.dir/src/main.obj" \
"CMakeFiles/pin.dir/src/Vector/Vector2d.obj" \
"CMakeFiles/pin.dir/src/Vector/Vector3d.obj"

# External object files for target pin
pin_EXTERNAL_OBJECTS =

pin.exe: CMakeFiles/pin.dir/src/main.obj
pin.exe: CMakeFiles/pin.dir/src/Vector/Vector2d.obj
pin.exe: CMakeFiles/pin.dir/src/Vector/Vector3d.obj
pin.exe: CMakeFiles/pin.dir/build.make
pin.exe: CMakeFiles/pin.dir/linklibs.rsp
pin.exe: CMakeFiles/pin.dir/objects1.rsp
pin.exe: CMakeFiles/pin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\pi\Desktop\Aurilius\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable pin.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\pin.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pin.dir/build: pin.exe

.PHONY : CMakeFiles/pin.dir/build

CMakeFiles/pin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\pin.dir\cmake_clean.cmake
.PHONY : CMakeFiles/pin.dir/clean

CMakeFiles/pin.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\pi\Desktop\Aurilius C:\Users\pi\Desktop\Aurilius C:\Users\pi\Desktop\Aurilius\cmake-build-debug C:\Users\pi\Desktop\Aurilius\cmake-build-debug C:\Users\pi\Desktop\Aurilius\cmake-build-debug\CMakeFiles\pin.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pin.dir/depend

