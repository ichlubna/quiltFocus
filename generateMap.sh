#!/bin/bash
# Parameters: input views, input depths, output file.hdr
# Input files need to be specified as ffmpeg input like directory/%04d.png
# The depth maps are in range 0-255, generated, for example, by:
# https://github.com/LiheYoung/Depth-Anything
# The depth is used as alpha in the input images
VIEWS=$(realpath $1)
DEPTHS=$(realpath $2)
TEMP=$(mktemp -d)
COMBINED=$TEMP/combined
mkdir $COMBINED
COUNT=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 $VIEWS)
cd build
ffmpeg -y -i $VIEWS -i $DEPTHS -filter_complex "[1:v]extractplanes=r[alf];[0:v][alf]alphamerge" $COMBINED/%04d.png
./quiltFocus -i $COMBINED -o $TEMP -rows $COUNT -cols 1
cd -
magick $TEMP/output.hdr -kuwahara 5 $TEMP/outputKuwahara.hdr
mv $TEMP/outputKuwahara.hdr $3
rm -rf $TEMP
COORDS=$(magick identify -precision 5 -define identify:locate=minimum -define identify:limit=3 test.hdr | grep Red: | sed 's/.*\ //')
DEPTHS=$(dirname $DEPTHS)
DEPTH_FILES=($DEPTHS/*)
echo "Potentially focused depth:"
magick ${DEPTH_FILES[0]} -crop +${COORDS/,/+} -format "%[fx:round(u.r*255)]" info:
echo ""
