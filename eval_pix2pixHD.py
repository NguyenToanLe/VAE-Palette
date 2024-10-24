import argparse
# from cleanfid import fid
# from core.base_dataset import BaseDataset
# from models.metric import inception_score
from models.metric import calculate_lpips_from_path_pix2pixHD, calculate_mse_score_from_path_pix2pixHD, calculate_ssim_score_from_path_pix2pixHD
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Input directory')
    parser.add_argument('-f', '--file', type=str, help='Output directory')

    ''' parser configs '''
    args = parser.parse_args()

    # fid_score = fid.compute_fid(args.src, args.dst)
    # is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)
    #
    # print('FID: {}'.format(fid_score))
    # print('IS:{} {}'.format(is_mean, is_std))

    lpips_scores, lpips_score, lpips_score_std = calculate_lpips_from_path_pix2pixHD(args.src, return_std=True,
                                                                                     return_list=True)
    mse_score, mse_score_std = calculate_mse_score_from_path_pix2pixHD(args.src, return_std=True)
    ssim_score, ssim_score_std = calculate_ssim_score_from_path_pix2pixHD(args.src, return_std=True)

    print('------------------------------Validation Start------------------------------')
    scores = f'Input directory: {args.src}\n'
    scores += "\n------------------------------Validation Start------------------------------\n"
    scores += f'LPIP Scores:\n'
    for img, score in lpips_scores.items():
        scores += f'{img}:   {score}\n'
    scores += f'LPIPS Score: {lpips_score}  +-  {lpips_score_std}\n\n'
    scores += f'MSE Score: {mse_score}  +-  {mse_score_std}\n\n'
    scores += f'SSIM Score: {ssim_score}  +-  {ssim_score_std}'
    scores += "\n------------------------------Validation End------------------------------\n\n"
    print('------------------------------Validation End------------------------------')

    log_name = os.path.join(args.file, 'loss.txt')
    with open(log_name, "w") as log_file:
        log_file.write('%s\n' % scores)

    print("Finish!!!!!")
