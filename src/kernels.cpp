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
    int Zc = 0, Zm = 0, Zn = 0;

    Zc = W->size[0]; // ASK: Is number of output channels same here as number of channels in wt and img
    Zn = X->size[1] - W->size[1] + 1;
    Zm = X->size[2] - W->size[2] + 1;
    int N = Z->size[0]; // number of filters
    int j = 0, k = 0;

    Tensor *currFilter = NULL; // TODO: check multiple filters handling
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
    float **size_x = X[0][0];
    //add right boundaries
    //for (size_t i = 0; i < len(size_x); i++)
    {
        /* code */
    }

    //out is tensor wwith 1 row
    
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
}
