CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
TARGET   := graph_evolve

.PHONY: all run json clean

all: $(TARGET)

$(TARGET): graph_evolve.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

json: $(TARGET)
	./$(TARGET) --json

clean:
	rm -f $(TARGET) *.o *.obj *.exe
