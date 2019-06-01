Compile and use PESQ for speech enhancement evaluation:

Download the PESQ source code from: https://www.itu.int/rec/T-REC-P.862-200102-I/en

Then compile:

$ gcc -o PESQ *.c -lm

Then use:

./pesq TEST_DR1_MRJO0_SI1364.WAV TEST_DR1_MRJO0_SI1364_dnnEnh.wav +16000
