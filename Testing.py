if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--image_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--image_format', type=str, required=True,  help='image format, png/jpg')
    parser.add_argument('--prefix', type=str, required=True, help='image prefix')
    parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    args = parser.parse_args()
    args.image_dir = "BiomechanicalAnalysisAruco"
    args.prefix = "Chessboard"
    args.image_format = "jpeg"
    args.square_size = 0.01
    args.width = 9
    args.height = 6
    ret, mtx, dist, rvecs, tvecs = calibrate(args.image_dir, args.prefix, args.image_format, args.square_size, args.width, args.height)