CC = nvcc
CFLAGS = -std=c++11
INCLUDES = 
LDFLAGS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
SOURCES = histogramEqualization.cu
OUTF = histogramEqualization.exe
OBJS = histogramEqualization.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)
