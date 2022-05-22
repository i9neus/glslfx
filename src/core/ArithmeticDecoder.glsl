/*//////////////////////////////////////////////////////////////////////////////////////////////////

   Copyright (c) 2021 Ben Spencer   
   Released under the MIT Licence

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.
   
    
   Description:
   
   This file is responsible for the texture decompression and internal state. The blocks of data
   in the buffers below contain three representations of the input image at different resolutions.
   Which resolution the decoder uses can be selected by modifying the kTextureResolution macro in
   the Common source file. Larger images mean a slower compilation time than smaller ones. Older systems
   are likely to struggle with the highest resolutions, so choose level 0 if compilation takes a long time

//////////////////////////////////////////////////////////////////////////////////////////////////*/

// Some global variables that are set at the start of each shading operation
int kImageSize = 0;
int kVisibleImageSize = 0;
int kBlockRowsCols = 0;
int kNumBlocks = 0;
int kNumPrecincts = 0;

// This function retrieves a specific encoded bit and increments the look-up indices accordingly
uvec3 UnpackBit(in uvec3 indices, uint inLength)
{
    // The bit index is one plus the index to be retrieved. If we're at bit zero then increment the index of the encoded uint
    // and reset the bit index to 32u.
    if (indices.y == 0u)
    {
        if (indices.x == inLength - 1u) { return uvec3(indices.xy, 0u); }
        indices.x++;
        indices.y = 32u;
    }
    
    // Unpack and return the indices and the data we want.
    //   x: The index of the current encoded uint
    //   y: The index of the current encoded bit + 1
    //   z: The retrived bit itself
    return uvec3(indices.x, indices.y - 1u, (encoded[indices.x] >> (indices.y - 1u)) & 1u);
}

/**
Fixed model arithmetic decoder
This function ingests the raw compressed stream and returns the quantised coefficient at a particular index in the block.
The serial nature of these decoders means that we're doing a ton of redunant processing to decode the image this way. 
However, since the framerate is locked to 60fps, it doesn't actually help us to pass the codec state between frames 
to reduce redundancy.
**/
uint DecodeMacroblock(ivec2 dataOffsets,           // The start and end of the encoded data
                      ivec2 modelOffsets,          // The stard and end of the model used to decode the data
                      int outElementIdx)           // The index of the 16-bit value we want to retrieve
{    
    // Prime the registers 
    uint f = encoded[dataOffsets.x];
    uint i0 = 0u;
    uint i1 = 0xffffffffu;
   
    // Move some indices into local variables for simplicty
    uvec3 bitIndices = uvec3(dataOffsets.x, 0u, 0u);
    uint inEncodedEnd = uint(dataOffsets.y);
    int modelStart = modelOffsets.x;
    int modelEnd = modelOffsets.y;

    for (int decodeIdx = 0; decodeIdx <= kBlockArea; decodeIdx++)
    {
        uint di = i1 - i0;
        uint diHi = di >> 16;
        uint diLo = di & 0xffffu;

        // Binary search for the element corresponding to the value contained in the register, f
        int j0 = modelStart, j1 = modelEnd - 1;
        while (j1 - j0 > 0)
        {
            int jMid = j0 + (j1 - j0) / 2;
            uint fMid = i0 + (diHi * (encoded[jMid] & 0xffffu)) + ((diLo * (encoded[jMid] & 0xffffu)) >> 16u);
            if (fMid <= f) { j0 = jMid + 1; }
            else { j1 = jMid; }
        }
        if (j1 == modelStart) { j1 = modelStart + 1; }

        // Retrieve the symbol from the model. Return if it's the one we're looking for.
        uint symbol = encoded[j1] >> 16u;
        if (decodeIdx == outElementIdx) 
        {   
            return symbol;
        }

        // Update the interval based upon the element in the model
        i1 = i0 + (diHi * (encoded[j1] & 0xffffu)) + ((diLo * (encoded[j1] & 0xffffu)) >> 16u);
        i0 = i0 + (diHi * (encoded[j1 - 1] & 0xffffu)) + ((diLo * (encoded[j1 - 1] & 0xffffu)) >> 16u);

        // If we're in an underflow state, do a bit of trickery to clear it
        if (((i1 >> 30u) & 3u) == 2u && ((i0 >> 30u) & 3u) == 1u)
        {
            int nudge;
            for (nudge = 0; nudge <= 29; nudge++)
            {
                uint bitmask = 1u << (29 - nudge);
                if ((i0 & bitmask) == 0u || (i1 & bitmask) != 0u) { break; }
            }
            if (nudge == 29) { return 0u; } // Would normally assert here, but we can't

            nudge++;

            // Update the registers to clear the underflow
            f = (f << nudge) ^ (1u << 31u);
            i0 = (i0 << nudge) ^ (1u << 31u);
            i1 = (i1 << nudge) ^ (1u << 31u);

            // Unpack some more bits
            for (int shift = nudge - 1; shift >= 0; shift--) 
            { 
                bitIndices = UnpackBit(bitIndices, inEncodedEnd);
                f |= bitIndices.z << shift; 
            }
        }
        // Not an underflow state, so just sweep off as many prefixing bits as we can
        else
        {
            int nudge;
            for (nudge = 0; nudge < 31; nudge++)
            {
                uint bitmask = 1u << (31 - nudge);
                if ((i0 & bitmask) != (i1 & bitmask)) { break; }
            }

            if (nudge > 0)
            {
                f <<= nudge;
                i0 <<= nudge;
                i1 <<= nudge;

                // Unpack some more bits
                for (int shift = nudge - 1; shift >= 0; shift--) 
                { 
                    bitIndices = UnpackBit(bitIndices, inEncodedEnd);
                    f |= bitIndices.z << shift; 
                }
            }
        }
    }

    return 0u; // Should never be here
}