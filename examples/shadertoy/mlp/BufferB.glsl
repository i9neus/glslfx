/*

NOTES:

Layer = Weights (N*N) -> Biases (N) -> Activations (N) -> Errors(N)
Network = Layer_1, ..., Layer_M, Input

Network weights are represented by 4-tuples with a spatial dimension of 2x2.  

4x4 matrix     

 0  1  2  3      x y x y      A   B
 4  5  6  7  ->  z w z w  -> 
 8  9 10 11      x y x y      C   D
12 13 14 15      z w z w 


*/

#define kLearningRate 1e-4
#define kMLPDebug false

#define ReLU(x)    (x * step(vec4(0.0), x))
#define DReLU(x)   step(vec4(0.0), x)

vec4 Tap(int idx)
{
    return texelFetch(iChannel0, ivec2(idx % int(iResolution.x), idx / int(iResolution.x)), 0);
}

vec4 Descent(vec4 n, vec4 dLdn, int netOffset)
{
    vec4 L = Tap(netOffset + kMLPLossOffset);
    return n - kLearningRate * L / (Sign(dLdn) * max(vec4(1e-6), abs(dLdn)));
}

int Forward(int netOffset, int layerIdx, int paramIdx, int frame, out vec4 B)
{
    if(layerIdx >= kMLPNumLayers || layerIdx != frame) { return kNotUpdated; }
    
    // Get the index of the block being evaluated. If it's not valid, bail out. 
    int actIdx = GetActIdx(paramIdx);
    if(actIdx < 0 || actIdx >= kMLPQuadsPerCol) 
    {       
        return kNotUpdated;
    }
    
    // Layer offset is the texel index of the start of this layer
    int layerOffset = netOffset + layerIdx * kMLPLayerStride;
     // Move the offset so it's pointing to the chunk associated with this activation quad   
    int weightsOffset = layerOffset + actIdx * kMLPRowStride * 2;
    // Compute the absolute offset to the start of the activation block (wind back one block from the current weights)
    int actOffset = layerOffset - (kMLPActStride + kMLPErrorStride);
    
    // Load the biases
    B = Tap(layerOffset + kMLPBiasOffset + actIdx);    
   
    for(int i = 0; i < kMLPQuadsPerCol; ++i, weightsOffset+=2)
    {        
        // Load the quad of evaluated parameters from the previous layer
        vec4 A = (layerIdx == 0) ? /*Tap(netOffset + kMLPInputOffset + actIdx)*/ vec4(1.0) : Tap(actOffset + i);
        
        vec4 M0 = Tap(weightsOffset);
        vec4 M1 = Tap(weightsOffset+1);
        vec4 M2 = Tap(weightsOffset+kMLPRowStride);
        vec4 M3 = Tap(weightsOffset+kMLPRowStride+1);
        
        B[0] += dot(vec4(M0[0], M0[1], M1[0], M1[1]), A);
        B[1] += dot(vec4(M0[2], M0[3], M1[2], M1[3]), A);
        B[2] += dot(vec4(M2[0], M2[1], M3[0], M3[1]), A);
        B[3] += dot(vec4(M2[2], M2[3], M3[2], M1[3]), A);      
    }
    
    // Activation function
    B = ReLU(B);

    return kUpdated;
}

int Backward(int netOffset, int layerIdx, int paramIdx, int texelIdx, int frame, int epoch, out vec4 B)
{    
    if(layerIdx >= kMLPNumLayers) { return kNotUpdated; }
    
    // Remap the clock tick so it advances backwards through the layers.
    frame = kMLPNumLayers - 1 - (frame - kBackwardFrame);
    
    // Update the weights and biases.
    // NOTE: The propagated error must first be primed, so only update the weights after the first epoch has completed
    if(layerIdx == frame && epoch > 0)
    {        
        // Weights
        if(IsWeight(paramIdx))
        {           
            ivec2 weightBlockPos = ivec2(paramIdx % kMLPRowStride, paramIdx / kMLPRowStride);
            int layerOffset = netOffset + layerIdx * kMLPLayerStride;
            
            // Load the 2x2 weight block
            B = Tap(texelIdx);
            // Load the activation values and the error values associated with the weight
            vec4 act = (layerIdx == 0) ? vec4(1.0) : Tap(PrevLayer(layerOffset) + kMLPActOffset + (weightBlockPos.x / 2));
            vec4 err = Tap(layerOffset + kMLPErrorOffset + (weightBlockPos.y / 2));
            
            // Calculate the partial derivative of loss with respect to each weight
            vec4 dLdw;
            weightBlockPos = (weightBlockPos % ivec2(2)) * 2;
            dLdw[0] = act[0 + weightBlockPos.x] * err[0 + weightBlockPos.y];
            dLdw[1] = act[1 + weightBlockPos.x] * err[0 + weightBlockPos.y];
            dLdw[2] = act[0 + weightBlockPos.x] * err[1 + weightBlockPos.y];
            dLdw[3] = act[1 + weightBlockPos.x] * err[1 + weightBlockPos.y];
            
            // Apply gradient descent
            B = Descent(B, dLdw, netOffset);
            
            if(kMLPDebug) { B = (paramIdx == 0) ? vec4(B.x, vec3(0.0)) : vec4(0.0); }
            
            return kUpdated;
        }
        // Update the biases
        else if(IsBias(paramIdx))
        {            
            // Load the bias
            B = Tap(texelIdx);
            
            // Estimate the partial derivative of the bias with respect to the loss. 
            vec4 dLdB = Tap(texelIdx + kMLPQuadsPerCol * 2);
            
            // Apply gradient descent
            B = Descent(B, dLdB, netOffset);
            
            if(kMLPDebug) { B = vec4(0.0); }
            
            return kUpdated;
        }        
    }
    // Propagate the error to the previous later
    else if(layerIdx == frame - 1)
    {
        int errorIdx = GetErrorIdx(paramIdx);
        if(errorIdx < 0 || errorIdx >= kMLPQuadsPerCol) 
        {       
            return kNotUpdated;
        }
        
        // NOTE: To save time, we propagate the error at the same time as updating the weights.
        // This shouldn't be a problem so long as the learning rate is relatively small.
        
        // Layer offset is the texel index of the start of this layer
        int layerOffset = netOffset + layerIdx * kMLPLayerStride;
         // Move the offset so it's pointing to the chunk associated with this activation quad   
        int weightsOffset = layerOffset + errorIdx * 2;
        // Compute the absolute offset to the start of the following layer's error block
        int errorOffset = layerOffset + kMLPLayerStride + kMLPErrorOffset;        

        B = vec4(0.0);        
        for(int x = 0; x < kMLPQuadsPerCol; ++x, weightsOffset+=2*kMLPRowStride)
        {        
            // Load the quad of evaluated parameters from the previous layer
            vec4 A = Tap(errorOffset + x);

            vec4 M0 = Tap(weightsOffset);
            vec4 M1 = Tap(weightsOffset+1);
            vec4 M2 = Tap(weightsOffset+kMLPRowStride);
            vec4 M3 = Tap(weightsOffset+kMLPRowStride+1);

            B[0] += dot(vec4(M0[0], M0[2], M2[0], M2[2]), A);
            B[1] += dot(vec4(M0[1], M0[3], M2[1], M2[3]), A);
            B[2] += dot(vec4(M1[0], M1[2], M3[0], M3[2]), A);
            B[3] += dot(vec4(M1[1], M1[3], M3[1], M3[3]), A);      
        }
        
        // Activation derivative
        B *= DReLU(Tap(layerOffset + kMLPActOffset));
        
        if(kMLPDebug) { B.yzw = vec3(0.0); }
        
        return kUpdated;
    }    
    
    return kNotUpdated;
}

int Loss(int netOffset, int layerIdx, int paramIdx, out vec4 B)
{
    // Average together the activations in the last layer
    int actOffset = netOffset + (kMLPNumLayers - 1) * kMLPLayerStride + kMLPActOffset;
    
    #define kDummyOut (kMLPDebug ? vec4(1.0, 0.0, 0.0, 0.0) : vec4(1.0))
    
    // For the loss layer, compute the L2 function
    if(layerIdx == kMLPOutputLayer)
    {
       B = vec4(0.0);
       for(int i = 0; i < kMLPQuadsPerCol; ++i)
       {
           B += dot(vec4(1.0), sqr(Tap(actOffset + i) - kDummyOut));
       }
       B /= float(kMLPWidth);
       
       return kUpdated;
    }
    
    // For the last layer, backprop the error from the loss
    else if(layerIdx == kMLPOutputLayer - 1)
    {
        // Get the activation index corresponding to the error index
        int actIdx = GetErrorIdx(paramIdx);
        if(actIdx < 0 || actIdx >= kMLPQuadsPerCol) { return kNotUpdated; }
        
        vec4 activation = Tap(actOffset + actIdx);
        B = 2.0 * (activation - kDummyOut) * DReLU(activation) / float(kMLPWidth);
        
        return kUpdated;
    }    
    
    return kNotUpdated;
}

vec4 Initialise(int netOffset, int layerIdx, int paramIdx)
{
    PCGInitialise(HashOf(uint(netOffset), uint(layerIdx), uint(paramIdx)));
    
    if(paramIdx >= kMLPBiasOffset) { return vec4(0.0); }
    
    vec4 B = (Rand() * 2.0 - 1.0) / (float(kMLPWidth) * 0.5);
    
    if(kMLPDebug)
    {
        if(IsBias(paramIdx)) 
        { 
            B.yzw = vec3(0.0);
        }
        if(IsWeight(paramIdx)) 
        {
            B = (paramIdx == 0) ? vec4(B.x, vec3(0.0)) : vec4(0.0);
        }
    }
    
    return B;
}
    

void mainImage( out vec4 rgba, in vec2 xy )
{    
    /*
        Clock schedule:
        0 -> 2: Forward propagation
        3:      Evaluate loss
        4 -> 6: Backward propagation
        7: Integrate parameters
        8: Update parameters
    */   
    
    // The index of the input texel
    int texelIdx = int(xy.y) * int(iResolution.x) + int(xy.x);
    // The index of the network in the batch
    int netIdx = texelIdx / kMLPNetStride;
    
    // The clock cycle that syncs the forward and backward propagation
    int time = (iFrame - 1) / kMLPFrameSkip;
    int frame = time % kFramesPerEpoch;
    int epoch = time / kFramesPerEpoch;
    
    if(int(xy.y) == int(iResolution.y) - 1)
    {
        if(int(xy.x) == 0 && epoch < int(iResolution.x) && frame == kMLPNumLayers + 2)
        {
            vec4 maxLoss = texelFetch(iChannel0, ivec2(0, int(iResolution.y) - 1), 0);
            vec4 newLoss = texelFetch(iChannel0, ivec2(epoch, int(iResolution.y) - 1), 0);
            
            rgba = max(newLoss, maxLoss);            
        }
        else if(int(xy.x) == epoch && frame == kMLPNumLayers + 1)
        {
            rgba = Tap(kMLPLossOffset);
        }
        else
        {
            rgba = texelFetch(iChannel0, ivec2(xy), 0);
        }
        return;
    }
    
    if(netIdx > 0) { return; }
    
    // The index of the layer associated with this texel
    int layerIdx = (texelIdx % kMLPNetStride) / kMLPLayerStride;    
    // The index of the parameter in the current layer
    int paramIdx = (texelIdx % kMLPNetStride) % kMLPLayerStride;   
    
    // On the first frame, initialise the parameters
    if(iFrame == 0)
    {
        rgba = Initialise(netIdx, layerIdx, paramIdx);
        return;
    }    
  
    int netOffset = netIdx * kMLPNetStride;    
    int state = kNotUpdated;
    // Propagate the data forward through the network
    if(frame < kMLPNumLayers)
    {
        state = Forward(netOffset, layerIdx, paramIdx, frame, rgba);     
    }
    // Update the loss function
    else if(frame == kMLPNumLayers)
    {
        state = Loss(netOffset, layerIdx, paramIdx, rgba);
    }
    // Backprop
    else 
    {        
        state = Backward(netOffset, layerIdx, paramIdx, texelIdx, frame, epoch, rgba); 
    }
    
    // If this texel wasn't updated, restore its original value from the buffer
    if(state == kNotUpdated) { rgba = texelFetch(iChannel0, ivec2(xy), 0);  }    
}