#include "kernels.h"
#include "stdio.h"
#include "stdlib.h"
#include "iostream"
using namespace std;
/*
 * Applies a 2d convolution on a 3D X using W: Z= W (conv) X + b
 * Tensor * X:		Input Tensor
 * Tensor * W:		Array of N weight Tensors (N == Z.size[0]) // feature extraction/ num of filters??
 * Tensor * Z:		Output Tensor
 * Tensor * b:		Bias
 */
void conv2d( Tensor *X, Tensor *W, Tensor *b, Tensor *Z )
{
    #if 1
    int Zc = 0, Zm = 0, Zn = 0;

    Zc = W->size[0]; // ASK: Is number of output channels same here as number of channels in wt and img
    Zn = X->size[1] - W->size[1] + 1;
    Zm = X->size[2] - W->size[2] + 1;
    int N = Z->size[0]; // number of filters
    int j = 0, k = 0;


    Tensor *currFilter = NULL; // TODO: check multiple filters handling

    for (size_t i = 0; i < (Z->size[0]*Z->size[1]*Z->size[2]); i++)
    {
        (*Z)[0][0][i] = 0;
    }
    


    for (size_t filters = 0; filters < N; filters++)
    {
        currFilter = &W[filters]; // ith weight tensor for ith output channel
        j = 0;
        k = 0;
        do
        {
            do
            {
                for (int c = 0; c < X->size[0]; c++)
                {
                    for (int p = 0; p < W->size[2]; p++)
                    {
                        for (int q = 0; q < W->size[1]; q++)
                        {
                            Z->data[filters][j][k] += (X->data[c][j + p][k + q]) * (currFilter->data[c][p][q]);
                        }
                    }
                }
                j++;
            } while (j < Zn);
            j = 0;
            k++;
        } while (k < Zm);
    }
    for (size_t i = 0; i < Z->size[0]; i++)
    {
        for (size_t j = 0; j < Z->size[1]; j++)
        {
            for (size_t k = 0; k < Z->size[2]; k++)
            {
                Z->data[i][j][k] += b->data[0][0][i];
            }
        }
    }
    #endif 

#if 0
     int size_input = X->size[1];
    int input_channels = X->size[0];
    int padding = 0;
    int stride = 1;
    int size_kernel = W->size[1];
    int kernel_amount = Z->size[0];

    //double
    
    
    int sum = 0;

    //output size of Z. Output size due to Conv.
    int size_output = size_input - size_kernel + 1; //(size_input + 2*padding - size_kernel)/stride + 1;
    Tensor* currW = NULL;
    //provide the CONV.
        //size_input ... orientation along the output. Get out of the output coordinates the input location via formula
    //channels output. Corresponds to one kernel in W
    for(int i = 0; i < kernel_amount; i++){
        //height
        currW =&W[i];
        for(int j = 0; j < size_output; j++){
            //width
            for(int l = 0; l < size_output; l++){
                
                //go to start position of current kernel location. Start from there considering all values from kernel.
                //for(int m = 0; m < kernel_size; m++){
                    //for(int n = 0; m < kernel_size; n++){
                        
                //forming the kernel. 
                //move inside input on location corresponding current location in output.
                //channels
                for(int c = 0; c < input_channels; c++){
                   //go to start position of current kernel location. Start from there considering all values from kernel.
                    for(int m = 0; m < size_kernel; m++){
                        for(int n = 0; n < size_kernel; n++){ 
                            //w*x ... compute the kernel sum(sum(sum)) for one position [j][l] on the input
                            //due to for(c) ... compute the sum over ALL channels of the input X for given position [j][l]
                            sum += (X->data[c][j+m][l+n])*(currW->data[c][m][n]);
                            }
                    }
                }
                //output value for kernel i on position [j][l] of the output matrix
                //next step: new position kernel [j][l+1] ... renew the value sum. Compute new kernel.
                Z->data[i][j][l] = sum;
                //printf("sum: %f ", sum);
                sum = 0;
            }
        }
    }
    //channels output
    for(int o=0; o < Z->size[0]; o++){
        //height of channel in Z
        for(int p = 0; p < Z->size[1]; p++){
            //width in channel in Z
            for(int q=0; q < Z->size[2]; q++){
                Z->data[o][p][q] += (b->data[0][0][o]);
            }
        }
    }
    #endif 
}


/*
 * Applies a max pool layer on X (size = stride = 2)
 * Tensor * X:	input Tensor
 * Tensor * Z:	output Tensor
 */
void maxPool(Tensor *X, Tensor *Z)
{
    int outSize = 0, stride_s = 2, pad_p = 0, size = 2;
    
    outSize = ((X->size[1] - size + (2 * pad_p) ) / stride_s) + 1;

    int  m = 0, n = 0, idxR = 0, idxC = 0;
    float max = 0;
    
    for (size_t outChnls = 0; outChnls < Z->size[0]; outChnls++)
    {
        idxC = 0;
        for (size_t i = 0; i < X->size[2]; i = i + stride_s )
        {
            idxR = 0;
            for (size_t j = 0; j < X->size[1]; j = j + stride_s )
            {
                max = -__FLT_MIN__; // or 0
                for ( m = 0; m < size; m++ )
                {
                    for ( n = 0; n < size; n++ )
                    {
                        if ( &(X->data[outChnls][m+j][n+i]) != NULL )
                        {
                            if ( ( X->data[outChnls][m+j][n+i] > max ) )
                            {
                                max = X->data[outChnls][m+j][n+i];
                            }
                        }
                    }
                }
                if ( idxR < outSize)
                {
                    Z->data[outChnls][idxR][idxC] = max;
                    idxR++;
                }
            }

            if ( idxC < outSize )
            {
                idxC++;                
            }
        }
    }
}

/*
 * Applies a Linear layer: z = Wx + b
 * Flatten the input if required
 * Tensor *	X: input Tensor
 * Tensor *	W: weight Matrix (in Tensor form)
 * Tensor *	B: bias array (in Tensor form)
 * Tensor *	Z: output array (in Tensor form)
 */

void Linear(Tensor *X, Tensor *W, Tensor *B, Tensor *Z)
{
    #if 0
    //    # Dot product y = w'x + b
    //      res = np.dot(self.w,x) + self.b

    float **size_x = X[0][0];
    //add right boundaries
    //for (size_t i = 0; i < len(size_x); i++)
    {
        Z->data[][] = X;
    }

    //out is tensor wwith 1 row
    #endif 
     //flattening the input tensor X.
    /*Method
        1. firstly go along the height coordinate [][h][]
        2. secondly, change to the width coordinate [][][w]. Change for one width element. Start over with [][h][].
        3. thirdly, if all height-width combinations are made ... change the channel [c][][]. Proceed with [][h][] and [][][w]
        */

    
    int input_channels = X->size[0];
    int input_height = X->size[1];
    int input_width = X->size[2];
    int tensor_elements = input_height*input_width*input_channels;
    


    double sum = 0;
    //flattened input X-->x
    Tensor x(1,1, tensor_elements);
    //cout<<x.size[0] * x.size[1] * x.size[2]<<endl;
    
    //flattening the input tensor X.
    /*Method
        1. firstly go along the width coordinate [][][w]
        2. secondly, change to the height coordinate [][h][]. Change for one height element. Start over with [][][w].
        3. thirdly, if all height-width combinations are made ... change the channel [c][][]. Proceed with [][][w] and [][h][].
        */
    //channels
    
    //if(X->size[0]!=1 && X->size[1]!=1){

        int index = 0;

        for(int i=0; i<X->size[0]; i++){
            /*//height
            for(int j=0; j< X->size[1]; j++){
                //width
                for(int k=0; k< X->size[2]; k++){
                    //flattening X to x
            
                    x.data[0][0][k + input_height*j + input_height*input_width*i] = X->data[i][j][k];
                }
            } */
            //height
            for(int j=0; j< X->size[1]; j++){
                //width
                for(int k=0; k< X->size[2]; k++){
                    //flattening X to x
                    int c = 0;
                    //if ( i == 0)
                    //  { c= 0; } else{ c = X->size[0];}
                        //x.data[0][0][j + j*X->size[1] + ((j*k )+i)] = X->data[i][j][k];
                        //x.data[0][0][index++] = X->data[i][j][k];
                        x.data[0][0][k + j*X->size[1] + (X->size[1]*X->size[2])*i] = X->data[i][j][k];
                }
            }
        }
    //}
    /* else{
        //if X has shape 1x1xVALUE, then transfer the values from input X to the flattened input version x.
        for(int i = 0; i < X->size[2]; i++){
            x.data[0][0][i] = X->data[0][0][i];
        } 
    }*/

    //Compute the different scalar products w.Tx and add bias b to the individual scalar products w.Tx+b.
    Tensor* currW = NULL;
    //Amount of weights. For each w one scalar product. 
    index = 0;
    
    //for (size_t i = 0; i < ; i++)
    //{
    //    /* code */
    //}
    
    for (size_t c = 0; c < W->size[0]; c++)
    {
        /* code */
        //currW = &W[c];
    
        for(int i = 0; i<W->size[1];i++){
            //Each weight element in currently considered weight vector. Weight vector element W.
            for(int j = 0; j< W->size[2]; j++){
                //sum += currW->data[c][i][j]*x.data[0][0][index++];
                //sum += currW->data[c][i][j]*x.data[0][0][j];
                sum += W->data[c][i][j]*x.data[0][0][j];
            }
        

            sum += B->data[0][0][i];
            //printf("Size_of_sum: %lu", sizeof(sum));
            Z->data[0][0][i] = sum;
            sum = 0;
        }

    }
    /*printf("Size flattened x: %d\n", x.size[2]);
    printf("Weight channel: %d\n", W->size[0]);
    printf("Bias amount: %d\n", B->size[2]);*/
/*
    printf("Input X: %dx%dx%d\n",X->size[0], X->size[1], X->size[2]);
    printf("Flattened input x: %dx%dx%d\n",x.size[0], x.size[1], x.size[2]);
    printf("Weight W: %dx%dx%d\n",W->size[0], W->size[1], W->size[2]);
    printf("Bias B: %dx%dx%d\n",B->size[0], B->size[1], B->size[2]);
    printf("Output Z: %dx%dx%d\n",Z->size[0], Z->size[1], Z->size[2]);
  */  
    //printf("Weight*Input: %d", W->data*X->data);
}

/*
 * Applies the ReLU activation function: Z = ReLU(X)
 * Tensor * X: input Tensor
 * Tensor * Z: output Tensor
 */
void ReLU(Tensor *X, Tensor *Z)
{
    // apply reLU to feature maps, ie., output of convolution layer
    int N = Z->size[0];
    for (size_t outChnls = 0; outChnls < N; outChnls++)
    {
        for (size_t i = 0; i < Z->size[1]; i++)
        {
            for (size_t j = 0; j < Z->size[2]; j++)
            {
                if (0 > X->data[outChnls][i][j])
                { // Z->data[outChnls][i][j]
                    //(*Z)[outChnls][i][j] = 0;
                    Z->data[outChnls][i][j] = 0;
                }
                else
                {
                    //(*Z)[outChnls][i][j]=(*X)[outChnls][i][j];
                    Z->data[outChnls][i][j] = X->data[outChnls][i][j];
                }
            }
        }
    }
}

/*
 * Applies the Softmax activation function z = exp(x_i)/sum(exp(x_j))
 * This is a stable Softmax implementation
 * Tensor * X: input vector in Tensor form
 * Tensor * Z: output vector in Tensor form
 */
void Softmax(Tensor *X, Tensor *Z)
{
    /*First Part: Building the denominator sum ... sum(exp(x_j))*/
    float sum_exp = 0;
    for(int i=0; i < X ->size[0]; i++){
        for(int j=0; j < X ->size[1]; j++){
            for(int k=0; k < X->size[2]; k++){
                float tmp = (*X)[i][j][k];
                /*compute the euler exponent of the tensor value*/
                tmp = exp(tmp);
                /*building the sum ... denominator of the probability function*/
                sum_exp = sum_exp + tmp;
            }
        }
    }
    for(int i=0; i < X ->size[0]; i++){
        for(int j=0; j < X ->size[1]; j++){
            for(int k=0; k < X->size[2]; k++){
                float tmp = (*X)[i][j][k];
                float val = 0;
                /*compute the euler exponent of the tensor value*/
                tmp = exp(tmp);
                /*compute the probability z = exp(x_i)/sum(exp(x_j))*/
                val = tmp / sum_exp;
                (*Z)[i][j][k] = val;
            }
        }
    }

}
