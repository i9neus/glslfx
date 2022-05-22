/*

MLP Solver

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

#define kPatchSampler iChannel1
#define kMLPSampler   iChannel0
#define kRenderSampler iChannel3

#define kL0 0
#define kL1 1
#define kL2 2
#define kLn 3
#define kLossFunction kL2

#define kLossAlpha 0.9

#define kMLPDebug false

#define Tap(idx) texelFetch(iChannel0, ivec2((idx) % int(iResolution.x), (idx) / int(iResolution.x)), 0)

vec4 MakeMask(int layerIdx, int paramIdx)
{
    return step(vec4(0.0), vec4(float(kMLPActiveNeurons[layerIdx] - paramIdx * 4)) - vec4(0.5, 1.5, 2.5, 3.5));
}

vec4 GetPatchState(int datasetIdx, int offset)
{
    return texelFetch(kPatchSampler, ivec2((datasetIdx * kPatchStateSize + offset) % int(iResolution.x), 
                                           (datasetIdx * kPatchStateSize + offset) / int(iResolution.x)), 0);   
}

vec4 GetPatchWeights(int inoutIdx)
{
    #if kKernelFilter == 0
        return vec4(1.0);
    #endif
    
    inoutIdx %= kMLPQuadsPerPatch;
    vec4 weights;
    for(int dim = 0; dim < 4; ++dim)
    {
        vec2 uv = vec2((inoutIdx * 4 + dim) % kMLPPatchSize, (inoutIdx * 4 + dim) / kMLPPatchSize) - vec2(kMLPPatchRadius);
        weights[dim] = 1.0 - sqr(length(uv) / float(kMLPPatchRadius + 1));
    }
    return weights;
}

vec4 LoadTestingData(int inoutIdx, int type)
{    
    vec4 L = 2.0 * PrepareInputPatch(ivec2(iMouse.xy), type, inoutIdx, kRenderSampler, iResolution.xy);
    
    return sign(L) * pow(abs(L), vec4((type == kMLPInput) ? kMLPInputGamma : kMLPOutputGamma)) * GetPatchWeights(inoutIdx);
}

vec4 LoadTrainingData(int datasetIdx, int inoutIdx, int type, bool doTest)
{         
    #if kMLPEnableInteractiveTesting == 1
        if(doTest)
        {
            return LoadTestingData(inoutIdx, type);
        }
    #endif
    
    if(GetPatchState(datasetIdx, 0).x == kPatchUnlocked) { return vec4(0.0); }
    
    int texelIdx;
    int patchIdx = datasetIdx * 3;
    int typeIdx = 0;
    float gamma;
    if(type == kMLPInput)
    {        
        typeIdx = inoutIdx / kMLPQuadsPerPatch;
        texelIdx = (inoutIdx * 4) % (kMLPWidth / 2);
        gamma = kMLPInputGamma;
    }
    else
    {
        typeIdx = 2;        
        texelIdx = inoutIdx * 4;
        gamma = kMLPOutputGamma;
    }   
    
    patchIdx += typeIdx;
    int kPatchesPerRow = int(iResolution.x) / kMLPPatchSize - 1;
    ivec2 texelXy = (ivec2(patchIdx % kPatchesPerRow, patchIdx / kPatchesPerRow)) * kMLPPatchSize + kPatchDataStride;     
    
    // Load the mean value of the patch from the state. The output patch should use the same mean as the input patch
    // since that's the only mean we have during inference. 
    float meanL = GetPatchState(datasetIdx, 1)[typeIdx % 2];
    
    vec4 L = vec4(0.0);
    for(int c = 0; c < 4 && texelIdx < kMLPPatchArea; ++c, ++texelIdx)
    {
        L[c] = kMLPPatchGain * (texelFetch(iChannel1, texelXy + ivec2(texelIdx % kMLPPatchSize, texelIdx / kMLPPatchSize), 0).x - meanL);
    }
    
    return sign(L) * pow(abs(L), vec4(gamma)) * GetPatchWeights(inoutIdx);
}

void SetBias(int layerIdx, int paramIdx, int targetLayer, int biasIdx, float value, inout vec4 quad)
{
    if(targetLayer != layerIdx) { return; }
    
    paramIdx = GetBiasIdx(paramIdx);
    
    if(paramIdx < 0 || paramIdx != biasIdx / 4) { return; }
    
    quad[biasIdx % 4] = value;   
}

void SetWeight(int layerIdx, int paramIdx, int targetLayer, ivec2 xy, float value, inout vec4 quad)
{
    if(targetLayer != layerIdx) { return; }
    
    ivec2 weightBlockPos = ivec2(paramIdx % kMLPRowStride, paramIdx / kMLPRowStride);
    if(weightBlockPos.x != xy.x / 2 || weightBlockPos.y != xy.y / 2) { return; }
    
    quad[2 * (xy.y % 2) + xy.x % 2] = value;   
}

vec4 Initialise(int netIdx, int layerIdx, int paramIdx)
{    
    if(IsGradientNet(netIdx) || !IsValidNet(netIdx)) { return vec4(.0); }
    
    if(IsRateNet(netIdx)) 
    {
        return vec4(0.7);
    }
    
    PCGInitialise(HashOf(uint(0), uint(layerIdx), uint(paramIdx)));
    
    if(paramIdx >= kMLPActOffset) { return vec4(0.0); }
    
    vec4 B;    
    if(IsBias(paramIdx)) 
    { 
        //B = mix(vec4(0.01), vec4(0.05), Rand());
        //B = vec4(0.5) + (Rand() * 2.0 - 1.0) * 0.1;
        B = vec4(0.0);
        //B *= 1.0 / float(kMLPActiveNeurons[layerIdx]);
        
        /*SetBias(layerIdx, paramIdx, 0, 0, -0.5, B);
        SetBias(layerIdx, paramIdx, 0, 1, -0.5, B);
        SetBias(layerIdx, paramIdx, 1, 0, 0.0, B);*/
    }
    if(IsWeight(paramIdx))
    {     
        ivec2 weightBlockPos = 2 * ivec2(paramIdx % kMLPRowStride, paramIdx / kMLPRowStride);     
        
        //B = vec4(0.5);
        B = Rand() * 2.0 - 1.0;        
        //B = (vec4(0.5) + (Rand() * 2.0 - 1.0) * 0.1);
        
        B *= 0.5;
        //B /= 0.25 * float(kMLPActiveNeurons[layerIdx]);
        
        
        /*B = vec4(0.0);
        SetWeight(layerIdx, paramIdx, 0, ivec2(14, 3), 1.0, B);
        SetWeight(layerIdx, paramIdx, 0, ivec2(2, 7), 1.0, B);
        SetWeight(layerIdx, paramIdx, 1, ivec2(3, 9), 1.0, B);
        SetWeight(layerIdx, paramIdx, 1, ivec2(3, 15), 1.0, B);
        SetWeight(layerIdx, paramIdx, 2, ivec2(9, 11), 1.0, B);
        SetWeight(layerIdx, paramIdx, 2, ivec2(1, 5), 1.0, B);*/
            
        // Drop any weights that are connected to inactive neurons
        if(weightBlockPos.x >= kMLPActiveNeurons[layerIdx]) { B.xz = vec2(0.0); } 
        if(weightBlockPos.x + 1 >= kMLPActiveNeurons[layerIdx]) { B.yw = vec2(0.0); } 
        if(weightBlockPos.y >= kMLPActiveNeurons[layerIdx+1]) { B.xy = vec2(0.0); } 
        if(weightBlockPos.y + 1 >= kMLPActiveNeurons[layerIdx+1]) { B.zw = vec2(0.0); } 
    }
    
    return B;
}

int Clear(int netOffset, int netIdx, int layerIdx, int paramIdx, inout MLPCtx ctx, inout vec4 B)
{
    if(ctx.frameIdx != 0) { return kNotUpdated; }
    
    // Clear the weights and biases of the batch nets and the entire gradient net
    if((IsBatchNet(netIdx) && IsWeightOrBias(layerIdx, paramIdx)) ||
       (IsGradientNet(netIdx) && ctx.iterationIdx == 0))
    {
        B = vec4(0.0);
        return kUpdated;
    }
        
    return kNotUpdated;
}

int Forward(int netOffset, int layerIdx, int paramIdx, int frame, inout vec4 B)
{    
    if(layerIdx != frame) { return kNotUpdated; }
    
    // Get the index of the block being evaluated. If it's not valid, bail out. 
    int actIdx = GetActIdx(paramIdx);
    if(actIdx < 0 || actIdx >= kMLPQuadsPerCol || actIdx * 4 >= kMLPActiveNeurons[layerIdx+1]) 
    {       
        return kNotUpdated;
    }
    
    // Layer offset is the texel index of the start of this layer
    int layerOffset = layerIdx * kMLPLayerStride;
     // Move the offset so it's pointing to the chunk associated with this activation quad   
    int weightsOffset = /*kMLPReferenceNet * kMLPNetStride + */layerOffset + actIdx * kMLPRowStride * 2;
    // Compute the absolute offset to the start of the activation block (wind back one block from the current weights)
    int actOffset = netOffset + layerOffset - kMLPLayerStride + kMLPActOffset;
   
    B = vec4(0.0);
    for(int i = 0; i < kMLPQuadsPerCol && i * 4 < kMLPActiveNeurons[layerIdx]; ++i, weightsOffset+=2)
    {  
        // Load the quad of evaluated parameters from the previous layer
        vec4 A = (layerIdx == 0) ? GetInput(netOffset, i) : ActivationFunction(Tap(actOffset + i), i);
        
        vec4 M0 = Tap(weightsOffset);
        vec4 M1 = Tap(weightsOffset+1);
        vec4 M2 = Tap(weightsOffset+kMLPRowStride);
        vec4 M3 = Tap(weightsOffset+kMLPRowStride+1);
        
        B[0] += dot(vec4(M0[0], M0[1], M1[0], M1[1]), A);
        B[1] += dot(vec4(M0[2], M0[3], M1[2], M1[3]), A);
        B[2] += dot(vec4(M2[0], M2[1], M3[0], M3[1]), A);
        B[3] += dot(vec4(M2[2], M2[3], M3[2], M3[3]), A);      
    }
    
    // Reweight the error to compensate for different numbers of neurons in each layer
    if(kMLPReweight)
    {
        B *= float(kMLPActiveNeurons[layerIdx+1]) / float(kMLPActiveNeurons[layerIdx]);
    }
    //B = vec4(0.0);
    // Load the biases
    B += Tap(layerOffset + kMLPBiasOffset + actIdx);
    
    // Activation function
    if(layerIdx != kMLPOutputLayer)
    {
        B = ActivationFunction(B, actIdx);
    }

    return kUpdated;
}

float Loss(int netOffset, int layerIdx, int paramIdx)
{
    // Average together the activations in the last layer
    int actOffset = netOffset + kMLPOutputLayer * kMLPLayerStride + kMLPActOffset;
   
    float L = 0.0;
    for(int i = 0; i < kMLPQuadsPerCol && i * 4 < kMLPActiveNeurons[kMLPOutputLayer+1]; ++i)
    {
        // Mask off inactive values so they're not included in the loss                
        #if kLossFunction == kL2
            // L2
            L += dot(MakeMask(kMLPOutputLayer+1, i), sqr(/*ActivationFunction*/(Tap(actOffset + i)) - GetOutput(netOffset, i)));
        #elif kLossFunction == kL1
            // L1
            L += dot(MakeMask(kMLPOutputLayer+1, i), abs(/*ActivationFunction*/(Tap(actOffset + i)) - GetOutput(netOffset, i)));          
        #elif kLossFunction == kLn
            // Ln
            L += dot(MakeMask(kMLPOutputLayer+1, i), pow(abs((Tap(actOffset + i)) - GetOutput(netOffset, i)), vec4(kLossAlpha)) );
        #endif 
    }    

    return L / float(kMLPActiveNeurons[kMLPOutputLayer+1]);
}

int Backward(int netOffset, int layerIdx, int paramIdx, int frame, int epoch, inout vec4 error)
{
     // Remap the clock tick so it advances backwards through the layers.
    frame = kMLPNumLayers - 1 - (frame - kBackwardFrame);
    
    // The output layer is handled differently. 
    if(frame == kMLPOutputLayer)
    {
         // For the loss layer
        if(layerIdx == kMLPLossLayer && IsLoss(paramIdx))
        {
            error = vec4(Loss(netOffset, layerIdx, paramIdx));
            return kUpdated;
        }    

        // Compute the error according to the loss function
        if(layerIdx == kMLPOutputLayer && IsError(paramIdx))
        {              
            // Get the activation index corresponding to the error index
            int paramIdx = GetErrorIdx(paramIdx);
            if(paramIdx >= kMLPActiveNeurons[kMLPOutputLayer+1]) { return kNotUpdated; }    
            
            // Average together the activations in the last layer
            int layerOffset = netOffset + kMLPOutputLayer * kMLPLayerStride;
        
            vec4 act = Tap(layerOffset + kMLPActOffset + paramIdx);             

            // Mask off inactive values
            #if kLossFunction == kL2
                // L2
                error = 2.0 * (/*ActivationFunction*/(act) - GetOutput(netOffset, paramIdx)) /* DActivationFunction(act)*/;
            #elif kLossFunction == kL1
                // L1
                error = sign(/*ActivationFunction*/(act) - GetOutput(netOffset, paramIdx)) /* DActivationFunction(act)*/;
            #elif kLossFunction == kLn
                // L1
                error = /*ActivationFunction*/(act) - GetOutput(netOffset, paramIdx) /* DActivationFunction(act)*/;
                error = sign(error) * kLossAlpha * pow(abs(error), vec4(kLossAlpha - 1.0));           
            #endif
            
            error *= MakeMask(kMLPOutputLayer+1, paramIdx);
        
            return kUpdated;
        }
        
        return kNotUpdated;
    } 
    
    if(layerIdx != frame || layerIdx > kMLPOutputLayer) { return kNotUpdated; }  
    
    int errorIdx;
    if(IsError(paramIdx))           { errorIdx = GetErrorIdx(paramIdx); }
    //else if(IsPreError(paramIdx))   { errorIdx = GetPreErrorIdx(paramIdx); }
    else                            { return kNotUpdated; }
    
    if(errorIdx * 4 >= kMLPActiveNeurons[layerIdx+1]) { return kNotUpdated; }
    
    // Get the offsets for where the data is coming from
    int targetLayerOffset = layerIdx * kMLPLayerStride;
    int sourceLayerOffset = IsError(paramIdx) ? (targetLayerOffset + kMLPLayerStride) : targetLayerOffset;        
    int weightsOffset = /*kMLPReferenceNet * kMLPNetStride + */sourceLayerOffset + errorIdx * 2;
    targetLayerOffset += netOffset;
    sourceLayerOffset += netOffset;

    error = vec4(0.0);        
    for(int x = 0; x < kMLPQuadsPerCol && x * 4 < kMLPActiveNeurons[layerIdx+1]; ++x, weightsOffset+=2*kMLPRowStride)
    {        
        // Load the quad of error values from the next layer
        vec4 A = IsError(paramIdx) ? Tap(sourceLayerOffset + kMLPErrorOffset + x) : vec4(1.0);
        
        A *= MakeMask(layerIdx+1, x);          

        vec4 M0 = Tap(weightsOffset);
        vec4 M1 = Tap(weightsOffset+1);
        vec4 M2 = Tap(weightsOffset+kMLPRowStride);
        vec4 M3 = Tap(weightsOffset+kMLPRowStride+1);            
        
        error[0] += dot(vec4(M0[0], M0[2], M2[0], M2[2]), A);
        error[1] += dot(vec4(M0[1], M0[3], M2[1], M2[3]), A);
        error[2] += dot(vec4(M1[0], M1[2], M3[0], M3[2]), A);
        error[3] += dot(vec4(M1[1], M1[3], M3[1], M3[3]), A);
    }    
    
    // Reweight the error to compensate for different numbers of neurons in each layer
    if(kMLPReweight)
    {
        error *= float(kMLPActiveNeurons[layerIdx+1]) / float(kMLPActiveNeurons[layerIdx+2]);
    }
    
    // The error 
    if(IsError(paramIdx))
    {
        // Activation derivative
        error *= DActivationFunction(Tap(targetLayerOffset + kMLPActOffset + errorIdx), errorIdx);
    }   
    
    return kUpdated;
}

int CalculateGradients(int netOffset, int layerIdx, int paramIdx, inout vec4 dLdx)
{    
    if(layerIdx >= kMLPNumLayers) { return kNotUpdated;  }
         
    // Weights
    if(IsWeight(paramIdx))
    {           
        //return kNotUpdated;
        
        paramIdx = GetWeightIdx(paramIdx);
        
        ivec2 weightBlockPos = 2 * ivec2(paramIdx % kMLPRowStride, paramIdx / kMLPRowStride);
        int layerOffset = netOffset + layerIdx * kMLPLayerStride;            
        
        if(weightBlockPos.x >= kMLPActiveNeurons[layerIdx] || weightBlockPos.y >= kMLPActiveNeurons[layerIdx+1])
        {
            dLdx = vec4(0.0);
            return kUpdated;
        }
        
        // Load the activation values and the error values associated with the weight
        vec4 act = (layerIdx == 0) ? GetInput(netOffset, weightBlockPos.x / 4) : 
                                     Tap(PrevLayer(layerOffset) + kMLPActOffset + weightBlockPos.x / 4);
        vec4 err = Tap(layerOffset + kMLPErrorOffset + (weightBlockPos.y / 4));
        
        // Calculate the partial derivative of loss with respect to each weight
        dLdx[0] = err[weightBlockPos.y % 4] * act[weightBlockPos.x % 4];
        dLdx[1] = err[weightBlockPos.y % 4] * act[weightBlockPos.x % 4 + 1];
        dLdx[2] = err[weightBlockPos.y % 4 + 1] * act[weightBlockPos.x % 4];
        dLdx[3] = err[weightBlockPos.y % 4 + 1] * act[weightBlockPos.x % 4 + 1];
        
        return kUpdated;
    }
    // Update the biases
    else if(IsBias(paramIdx))
    {            
        //return kNotUpdated;
        
        paramIdx = GetBiasIdx(paramIdx);
        
        // Estimate the partial derivative of the bias with respect to the loss. 
        dLdx = Tap(netOffset + layerIdx * kMLPLayerStride + kMLPErrorOffset + paramIdx);// * 
                    //Sign(Tap(netOffset + layerIdx * kMLPLayerStride + kMLPBiasOffset + paramIdx));
                    
        // Attenuate bias gradient
        //dLdx /= float(kMLPActiveNeurons[layerIdx]);

        return kUpdated;
    }
    
    return kNotUpdated;
}

int Integrate(int netIdx, int layerIdx, int paramIdx, ivec2 xyTexel, inout vec4 dLdx)
{      
    dLdx = vec4(0.0);
    for(int idx = kMLPAccumNets; idx < kMLPAccumNets + kMLPTrainSize; ++idx)
    {  
        dLdx += Tap(GetParamOffset(idx, layerIdx, paramIdx));
    }
    dLdx /= float(kMLPTrainSize); 
    
    // Add the pre-accumulated values
    dLdx += texelFetch(kMLPSampler, xyTexel, 0);

    return kUpdated;    
}

int Descend(int netIdx, int layerIdx, int paramIdx, int iterationsPerEpoch, inout vec4 B)
{      
    if(layerIdx > kMLPOutputLayer) { return kNotUpdated; }
        
    // Ignore anything that's not a weight or a bias
    int paramOffset;
    if(IsWeight(paramIdx))    { paramOffset = kMLPWeightOffset + GetWeightIdx(paramIdx);  }
    else if(IsBias(paramIdx)) { paramOffset = kMLPBiasOffset + GetBiasIdx(paramIdx); }
    else                      { return kNotUpdated; }   
    
    // Load the parameter and its derivative
    vec4 x = Tap(GetParamOffset(kMLPReferenceNet, layerIdx, paramOffset));
    vec4 dLdx = Tap(GetParamOffset(kMLPGradientNet, layerIdx, paramOffset)) / float(iterationsPerEpoch);            
    
    // Establish the learning rate
    vec4 paramRate = pow(vec4(10.0), mix(vec4(kMLPLearningRateMax), vec4(kMLPLearningRateMin),
                                     Tap(GetParamOffset(kMLPRateNet, layerIdx, paramOffset))));
                                     
    /*ivec2 weightBlockPos = 2 * ivec2(paramIdx % kMLPRowStride, paramIdx / kMLPRowStride);    
    paramRate[0] *= mix(ActivationAmplitudes(weightBlockPos.y / 4)[weightBlockPos.y % 4], 
                        ActivationAmplitudes(weightBlockPos.x / 4)[weightBlockPos.x % 4], 0.5);
    paramRate[1] *= mix(ActivationAmplitudes(weightBlockPos.y / 4)[weightBlockPos.y % 4], 
                        ActivationAmplitudes(weightBlockPos.x / 4)[weightBlockPos.x % 4 + 1], 0.5);
    paramRate[2] *= mix(ActivationAmplitudes(weightBlockPos.y / 4)[weightBlockPos.y % 4 + 1], 
                        ActivationAmplitudes(weightBlockPos.x / 4)[weightBlockPos.x % 4], 0.5);
    paramRate[3] *= mix(ActivationAmplitudes(weightBlockPos.y / 4)[weightBlockPos.y % 4 + 1], 
                        ActivationAmplitudes(weightBlockPos.x / 4)[weightBlockPos.x % 4 + 1], 0.5);*/
                        
    dLdx = sign(dLdx) * min(abs(dLdx), vec4(1e-0));
    
    // Apply gradient descent
    B = x - dLdx * paramRate;// * ActivationAmplitudes(paramIdx);
    
    
    return kUpdated;   
} 

int Propagate(int netIdx, int layerIdx, int paramIdx, inout vec4 B)
{
    if(!IsBatchNet(netIdx)) { return kNotUpdated; }
    
    if(layerIdx > kMLPOutputLayer || !(IsWeight(paramIdx) || IsBias(paramIdx))) { return kNotUpdated; }
    
    // Propagate the parameter from the reference net to the batch nets
    B = Tap(GetParamOffset(kMLPReferenceNet, layerIdx, paramIdx));
    
    return kUpdated;
}

int LoadData(int netIdx, int layerIdx, int paramIdx, int iterationIdx, out vec4 B)
{
    if(layerIdx != kMLPLossLayer) { return kNotUpdated; }
    
    // Get the index of the block to load based on the current iteration and the index of the network in the batch
    int blockIdx = IsReferenceNet(netIdx) ? 
                   int(float(kMLPTrainSize) * iMouse.x / iResolution.x) : 
                   ((netIdx - kMLPAccumNets) + iterationIdx * kMLPTrainSize);
        
    vec4 mask;
    int type;
    if(IsInput(paramIdx))
    {   
        paramIdx = GetInputIdx(paramIdx);
        mask = MakeMask(0, paramIdx);
        type = kMLPInput;
    }
    else if(IsOutput(paramIdx))
    {   
        paramIdx = GetOutputIdx(paramIdx);
        mask = MakeMask(kMLPNumLayers, paramIdx);
        type = kMLPOutput;
    }
    else
    {
       return kNotUpdated;
    }
    
    // Load the training data 
    B = LoadTrainingData(blockIdx, paramIdx, type, IsReferenceNet(netIdx)) * mask;
    return kUpdated;
}

bool LogLoss(vec2 xy, inout MLPCtx ctx, out vec4 rgba)
{
    if(int(xy.y) != int(iResolution.y) - 1) { return false; }    
    
    if(int(xy.x) == 0 && !ctx.doEstimateGradients)
    {
        float maxLoss = texelFetch(kMLPSampler, ivec2(0, int(iResolution.y) - 1), 0).x;
        float newLoss = texelFetch(kMLPSampler, ivec2((1 + (ctx.epochIdx - 1) / 4) % int(iResolution.x), int(iResolution.y) - 1), 0)[ctx.epochIdx%4];
            
        rgba = vec4(0.0);
        //rgba.x = max(newLoss, maxLoss);
        rgba.x = (ctx.epochIdx < 5) ? min(maxLoss, newLoss) : max(newLoss, mix(newLoss, maxLoss, 0.995));
    }
    else if(int(xy.x) - 1 == ctx.epochIdx / 4 && !ctx.doEstimateGradients)
    {
        float sumLoss = Tap(GetParamOffset(kMLPGradientNet, kMLPLossLayer, kMLPLossOffset)).x / float(ctx.iterationsPerEpoch);
        float sumWeights = 1.0;
        /*for(int i = 0; i < 0 && ctx.epochIdx - 1 - i >= 1; ++i)
        {
            float weight = 1.0;//exp(-float(i) * 0.1);            
            sumLoss += texelFetch(iChannel0, ivec2((1 + (ctx.epochIdx - 1 - i) / 4) % int(iResolution.x), int(iResolution.y) - 1), 0)[(ctx.epochIdx - 1 - i)%4] * weight;
            sumWeights += weight;
        }*/
        rgba = texelFetch(kMLPSampler, ivec2(xy), 0);
        rgba[ctx.epochIdx % 4] = sumLoss / sumWeights;
    }
    else
    {
        rgba = texelFetch(kMLPSampler, ivec2(xy), 0);
    }
    return true;
}

int UpdateLearningRates(int netIdx, int layerIdx, int paramIdx, inout MLPCtx ctx, inout vec4 lr)
{
    if(!IsWeight(paramIdx) && !IsBias(paramIdx)) { return kNotUpdated; }
    
    // Multiply the gradients of the two most recent time steps
    vec4 signdLdx = Tap(GetParamOffset(kMLPLastGradientNet, layerIdx, paramIdx)) *
                    Tap(GetParamOffset(kMLPGradientNet, layerIdx, paramIdx)) / float(ctx.iterationsPerEpoch);    
    
    // Get the current learning rate
    lr = Tap(GetParamOffset(kMLPRateNet, layerIdx, paramIdx));
    
    // Turn up the learning at a constant rate
    lr *= 1.0 - kMLPLearningRateUp;    

    // If the sign has switched, turn down the rate
    for(int i = 0; i < 4; ++i)
    {
        if(signdLdx[i] < 0.0)
        {
            lr[i] += (1.0 - lr[i]) * kMLPLearningRateDown;
        }
    }
    
    return kUpdated;
}

void mainImage( out vec4 rgba, in vec2 xy )
{    
    /*vec2 xi = xy / iResolution.xy;
    ivec2 uv = ivec2(xi * vec2(iResolution.x * 0.5, iResolution.y) + vec2(float(1) * iResolution.x * 0.5, 0.0));

    rgba = saturate(texelFetch(iChannel1, uv, 0));
    return;*/
    
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
    // The index of the layer associated with this texel
    int layerIdx = (texelIdx % kMLPNetStride) / kMLPLayerStride;    
    // The index of the parameter in the current layer
    int paramIdx = (texelIdx % kMLPNetStride) % kMLPLayerStride;   
    
     // On the first frame, initialise the parameters and pre-load the data
    if(iFrame == kMLPInitFrame)
    {
        rgba = Initialise(netIdx, layerIdx, paramIdx);
        LoadData(netIdx, layerIdx, paramIdx, 0, rgba);
        return;
    } 
    // Only update once per kMLPFrameSkip frames.
    else if(/*iFrame > 10 || */iFrame < kMLPStartFrame || (iFrame - kMLPStartFrame) % (kMLPFrameSkip + 1) != 0)
    { 
        rgba = texelFetch(iChannel0, ivec2(xy), 0); 
        return; 
    }     
    
    // Generate a context
    MLPCtx ctx = GetMLPIndices(iFrame, iResolution.xy);  
    
    // Once all gradients have been computed, log the loss
    if(LogLoss(xy, ctx, rgba)) 
    { 
        return; 
    }      
    
    // Only process the specified number of nets
    if(netIdx >= kMLPBatchSize + kMLPAccumNets) { return; }    
   
    int netOffset = netIdx * kMLPNetStride;    
    int state = kNotUpdated;
    rgba = vec4(0.0);
    
    // If we're estimating gradients, perform forward and backward passes    
    if(ctx.doEstimateGradients)
    {
        // Propagate the data forward through the network
        if(IsForwardFrame(ctx.frameIdx))
        {
            if(IsBatchNet(netIdx) || IsReferenceNet(netIdx))
            {
                state = Forward(netOffset, layerIdx, paramIdx, ctx.frameIdx, rgba);               
            }
            
            // Clear any accumulators
            state |= Clear(netOffset, netIdx, layerIdx, paramIdx, ctx, rgba);
        }
        // Propagate the error backward through the network
        else if(IsBackwardFrame(ctx.frameIdx))
        {        
            if(IsBatchNet(netIdx))
            {
                state = Backward(netOffset, layerIdx, paramIdx, ctx.frameIdx, ctx.epochIdx, rgba); 
            }
        }
        // Estimate noisy gradients
        else if(IsGradientFrame(ctx.frameIdx))
        {        
            if(IsBatchNet(netIdx))
            {
                state = CalculateGradients(netOffset, layerIdx, paramIdx, rgba);
            }
        }  
        // Integrate the gradients
        else if(IsIntegrateFrame(ctx.frameIdx))
        {
            if(IsGradientNet(netIdx))
            {
                state = Integrate(netOffset, layerIdx, paramIdx, ivec2(xy), rgba);
            }
            else if(IsBatchNet(netIdx) || IsReferenceNet(netIdx))
            {
                // Load data for the next round
                state = LoadData(netIdx, layerIdx, paramIdx, (ctx.iterationIdx + 1) % ctx.iterationsPerEpoch, rgba);
            }
        }
    }
    // Otherwise, update the weights and stats
    else
    {    
        if(IsRateUpdateFrame(ctx.frameIdx))
        {
            if(netIdx == kMLPRateNet)
            {
                state = UpdateLearningRates(netIdx, layerIdx, paramIdx, ctx, rgba);                
            }
        }
        else if(IsDescendFrame(ctx.frameIdx))
        {  
            if(netIdx == kMLPReferenceNet)
            {
                // Apply gradient descent
                state = Descend(netIdx, layerIdx, paramIdx, ctx.iterationsPerEpoch, rgba);
            }            
            else if(netIdx == kMLPLastGradientNet)
            {
                // Copy the gradients
                rgba = Tap(GetParamOffset(kMLPGradientNet, layerIdx, paramIdx)) / float(ctx.iterationsPerEpoch);
                state = kUpdated;
            }
        }        
    }
    
    // If this texel wasn't updated, restore its original value from the buffer
    if(state == kNotUpdated) { rgba = texelFetch(iChannel0, ivec2(xy), 0);  }  
}