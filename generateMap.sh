#!/bin/bash
#set -x
#set -e
# Parameters: input views, input depths, output file.hdr
# Input files need to be specified as ffmpeg input like directory/%04d.png
# The depth maps are in range 0-255, generated, for example, by:
# https://github.com/LiheYoung/Depth-Anything
# The depth is used as alpha in the input images
VIEWS=$(realpath $1)
DEPTHS=$(realpath $2)
TEMP=$(mktemp -d)

#mkdir -p build
cd build
#cmake ..
make
cd -

startTime=`date +%s.%N`
COMBINED=$TEMP/combined
mkdir $COMBINED
COUNT=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 $VIEWS)
cd build
# Combining depth maps with the input images as alpha channel
ffmpeg -y -i $VIEWS -i $DEPTHS -filter_complex "[1:v]extractplanes=r[alf];[0:v][alf]alphamerge" $COMBINED/%04d.png
# Computing the focus map
./quiltFocus -i $COMBINED -o $TEMP -rows $COUNT -cols 1 -l
cd -
# Applying Kuwahara filter to remove outliers
#magick $TEMP/output.hdr -morphology dilate octagon:1 $TEMP/output.hdr
magick $TEMP/output.hdr -kuwahara 1 $TEMP/output.hdr
mv $TEMP/output.hdr $3
endTime=`date +%s.%N`
echo "Time of focus map:"
echo $( echo "$endTime - $startTime" | bc -l )

# Blending the depth maps
startTime=`date +%s.%N`
NAMES=$(find $(dirname $DEPTHS) -maxdepth 1 | tail -n +2 | tr '\n' ' ')
ALL_BLEND_DEPTH=$TEMP/allBlendDepth.png 
magick $NAMES -evaluate-sequence Mean $ALL_BLEND_DEPTH

# Getting the coordinates of the minimum value
COORDS=$(magick identify -precision 5 -define identify:locate=minimum -define identify:limit=3 $3 | grep Red: | sed 's/.*\ //')
# Getting the minimum value
MINIMUM=$(magick $3 -crop +${COORDS/,/+} -format "%[fx:u.r]" info:)
TOLERANCE=300
# Getting the average depth from the pixels close to minimum
MINIMUM=$(bc <<< "scale=5; $MINIMUM*100+$TOLERANCE") 
magick $3 -color-threshold "gray(0%)-gray($MINIMUM%)" $TEMP/mask.png
magick $ALL_BLEND_DEPTH -alpha on \( +clone -channel a -fx 0 \) +swap $TEMP/mask.png -composite $TEMP/maskedDepth.png
DEPTH=$(magick $TEMP/maskedDepth.png -resize 1x1! -alpha off -depth 8 -format "%[pixel:p{0,0}]" info:)
DEPTH=$(echo $DEPTH | awk -F[\(\)] '{print $2}')
##DEPTH=$(bc <<< "scale=4; ($DEPTH/100)*255")

START=$(magick identify -precision 5 -define identify:locate=minimum -define identify:limit=3 $ALL_BLEND_DEPTH | grep Gray: | cut -d "(" -f2 | cut -d ")" -f1)
END=$(magick identify -precision 5 -define identify:locate=maximum -define identify:limit=3 $ALL_BLEND_DEPTH | grep Gray: | cut -d "(" -f2 | cut -d ")" -f1)
DEPTH_NOR=$(bc <<< "scale=5; ($DEPTH-255*$START)/(255*$END-255*$START)")
endTime=`date +%s.%N`
echo "Time of processing:"
echo $( echo "$endTime - $startTime" | bc -l )

echo "Potentially focused depth, normalized depth, at coords:"
echo $DEPTH
echo $DEPTH_NOR
echo $COORDS
rm -rf $TEMP
