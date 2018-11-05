### openauto
Open Auto, smart car, smart road, and et al.

### Requirements
- Python 2.7
- Cmake
- Swig, http://www.swig.org/

### Usage

0. mkdir build
1. cd build/
2. cmake ..
3. make
4. cpack --config CPackConfig.cmake
5. cpack --config CPackSourceConfig.cmake
6. run in c++
```
./facelock_exe
```
7. run in python
```
python -c "import facelock;facelock.facelock()"
```
