cd ..

for t in 1 2 3
do

  python main.py cifar10 student --moonshine --conv DConv   -t wrn_40_2_$t -s wrn_40_2_$t.DConv   --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 0 &
  python main.py cifar10 student --moonshine --conv DConvG2 -t wrn_40_2_$t -s wrn_40_2_$t.DConvG2 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 1 &
  python main.py cifar10 student --moonshine --conv DConvG4 -t wrn_40_2_$t -s wrn_40_2_$t.DConvG4 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 2 &
  python main.py cifar10 student --moonshine --conv DConvG8 -t wrn_40_2_$t -s wrn_40_2_$t.DConvG8 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 3

  python main.py cifar10 student --moonshine --conv DConvG16 -t wrn_40_2_$t -s wrn_40_2_$t.DConvG16 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 0 &
  python main.py cifar10 student --moonshine --conv DConvA16 -t wrn_40_2_$t -s wrn_40_2_$t.DConvA16 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 1 &
  python main.py cifar10 student --moonshine --conv DConvA8  -t wrn_40_2_$t -s wrn_40_2_$t.DConvA8  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 2 &
  python main.py cifar10 student --moonshine --conv DConvA4  -t wrn_40_2_$t -s wrn_40_2_$t.DConvA4  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 3

  python main.py cifar10 student --moonshine --conv A2B2  -t wrn_40_2_$t -s wrn_40_2_$t.A2B2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 0 &
  python main.py cifar10 student --moonshine --conv A4B2  -t wrn_40_2_$t -s wrn_40_2_$t.A4B2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 1 &
  python main.py cifar10 student --moonshine --conv A8B2  -t wrn_40_2_$t -s wrn_40_2_$t.A8B2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 2 &
  python main.py cifar10 student --moonshine --conv A16B2 -t wrn_40_2_$t -s wrn_40_2_$t.A16B2 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 3

  python main.py cifar10 student --moonshine --conv G16B2 -t wrn_40_2_$t -s wrn_40_2_$t.G16B2 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 0 &
  python main.py cifar10 student --moonshine --conv G8B2  -t wrn_40_2_$t -s wrn_40_2_$t.G8B2   --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 1 &
  python main.py cifar10 student --moonshine --conv G4B2  -t wrn_40_2_$t -s wrn_40_2_$t.G4B2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 2 &
  python main.py cifar10 student --moonshine --conv G2B2  -t wrn_40_2_$t -s wrn_40_2_$t.G2B2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 3

  python main.py cifar10 student --moonshine --conv ConvB2  -t wrn_40_2_$t -s wrn_40_2_$t.ConvB2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 0 &
  python main.py cifar10 student --moonshine --conv ConvB4  -t wrn_40_2_$t -s wrn_40_2_$t.ConvB4  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 1 &
  python main.py cifar10 student --moonshine --conv DConvB2  -t wrn_40_2_$t -s wrn_40_2_$t.DConvB2  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 2 &
  python main.py cifar10 student --moonshine --conv DConvB4  -t wrn_40_2_$t -s wrn_40_2_$t.DConvB4  --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 3

  python main.py cifar10 student --moonshine --conv Conv -t wrn_40_2_$t -s wrn_16_2_$t --wrn_depth 40 --wrn_width 2 --diff_shape --stu_depth 16 --stu_width 2 --cifar_loc='../data' --GPU 0 &
  python main.py cifar10 student --moonshine --conv Conv -t wrn_40_2_$t -s wrn_16_1_$t --wrn_depth 40 --wrn_width 2 --diff_shape --stu_depth 16 --stu_width 1 --cifar_loc='../data' --GPU 1 &
  python main.py cifar10 student --moonshine --conv DConvA2 -t wrn_40_2_$t -s wrn_40_2_$t.DConvA2 --wrn_depth 40 --wrn_width 2 --cifar_loc='../data' --GPU 2 &
  python main.py cifar10 student --moonshine --conv Conv -t wrn_40_2_$t -s wrn_40_1_$t --wrn_depth 40 --wrn_width 2 --diff_shape --stu_depth 40 --stu_width 1 --cifar_loc='../data' --GPU 3
done


