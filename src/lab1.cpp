#include "kernels.h"
#include "../utils/tensor.h"
#include <vector>




Tensor * readConv(Tensor * X, Tensor *B , Tensor * Ref, FILE * f);
void testConv(const char * infile);
void testLinear(const char * infile);
void testPool(const char * infile);
void testSoftmax(const char * infile);
void testReLU(const char * infile);



int main(int argc , char * argv[])
{
	//testConv("data/conv_test.dat");
//	testLinear("data/linear_test.dat");
	testPool("data/pool_test.dat");
//	testSoftmax("data/softmax_test.dat");
	testReLU("data/relu_test.dat");

	return 0;
}


void testConv(const char * infile)
{
	FILE * f = fopen(infile,"rb");
	Tensor X,R,B;
	printf("------------------------------\n");
	printf("Testing Convolutional Layer...\n");
	while(1){
		Tensor * W = readConv(&X,&R,&B,f);
		if(W == NULL)
			break;
		Tensor Z(R.size[0],R.size[1],R.size[2]);
		conv2d(&X,W,&B,&Z);
		compareTensors(&Z,&R,1,0.001);
		delete [] W;
	}
	fclose(f);
}

void testLinear(const char * infile)
{
	printf("------------------------------\n");
	printf("Testing Linear Layer...\n");
	FILE * f = fopen(infile,"rb");
	Tensor X,W,B,Ref;
	Tensor * tp[4] = {&X,&Ref,&W,&B};
	while(1){
		for(int i = 0; i < 4; i++){
			if(tp[i]->read(f) == TENSOR_READ_FAILED){
				fclose(f);
				return;
			}
		}
		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
		Linear(&X, &W , &B , &Z);
		compareTensors(&Z,&Ref, 1, 0.001);
	}
}

void testPool(const char * infile)
{
	FILE * f = fopen(infile,"rb");
	printf("------------------------------\n");
	printf("Testing Pool Layer...\n");
	Tensor X,Ref;
	while(1){
		if(X.read(f) == TENSOR_READ_FAILED){
			fclose(f);
			return;
		}
		if(Ref.read(f) == TENSOR_READ_FAILED){
			fclose(f);
			return;
		}
		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
		maxPool(&X,&Z);
		compareTensors(&Z, &Ref , 1, 0.001);
	}
}


void testSoftmax(const char * infile)
{
	printf("------------------------------\n");
	printf("Testing Softmax Layer...\n");
	FILE * f = fopen(infile,"rb");
	Tensor X,Ref;
	Tensor * tp[2] = {&X,&Ref};
	while(1){
		for(int i = 0; i < 2; i++){
			if(tp[i]->read(f) == TENSOR_READ_FAILED){
				fclose(f);
				return;
			}
		}
		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
		Softmax(&X,&Z);
		compareTensors(&Z,&Ref, 1, 0.001);
	}
}


void testReLU(const char * infile)
{
	printf("------------------------------\n");
	printf("Testing ReLU Layer...\n");
	FILE * f = fopen(infile,"rb");
	Tensor X,Ref;
	Tensor * tp[2] = {&X,&Ref};
	while(1){
		for(int i = 0; i < 2; i++){
			if(tp[i]->read(f) == TENSOR_READ_FAILED){
				fclose(f);
				return;
			}
		}
		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
		ReLU(&X,&Z);
		compareTensors(&Z,&Ref, 1, 0.001);
	}
}


Tensor * readConv(Tensor * X, Tensor * Ref, Tensor * B , FILE * f)
{
	if(X->read(f) == TENSOR_READ_FAILED)
		return NULL;
	if(Ref->read(f) == TENSOR_READ_FAILED)
		return NULL;
	// For multiple output channels we need a weight 
	// Tensor for every output feature map!
	Tensor * W = new Tensor[Ref->size[0]]();
	for(int i = 0; i < Ref->size[0] ; i++){
		if(W[i].read(f) == TENSOR_READ_FAILED){
			delete [] W;
			return NULL;
		}
	}
	if(B->read(f) == TENSOR_READ_FAILED){
		delete [] W;
		return NULL;
	}
	return W;
}

