import argparse
# from cleanfid import fid
# from core.base_dataset import BaseDataset
# from models.metric import inception_score
from models.metric import calculate_lpips_from_path, calculate_mse_score_from_path, calculate_ssim_score_from_path
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generated images directory')
    parser.add_argument('-f', '--file', type=str, help='Output directory')

    ''' parser configs '''
    args = parser.parse_args()

    # fid_score = fid.compute_fid(args.src, args.dst)
    # is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=True, batch_size=8, resize=True, splits=10)
    #
    # print('FID: {}'.format(fid_score))
    # print('IS:{} {}'.format(is_mean, is_std))

    lpips_scores, lpips_score, lpips_score_std = calculate_lpips_from_path(args.src, args.dst,
                                                                           return_list=True, return_std=True)
    mse_score, mse_score_std = calculate_mse_score_from_path(args.src, args.dst, return_std=True)
    ssim_score, ssim_score_std = calculate_ssim_score_from_path(args.src, args.dst, return_std=True)

    print('------------------------------Validation Start------------------------------')
    scores = f'GT directory: {args.src}\n'
    scores += f'Generated images directory: {args.dst}\n'
    scores += "\n------------------------------Validation Start------------------------------\n"
    scores += f'LPIP Scores:\n'
    for img, score in lpips_scores.items():
        scores += f'{img}:   {score}\n'
    scores += f'LPIPS Score: {lpips_score}  +-  {lpips_score_std}\n\n'
    scores += f'MSE Score: {mse_score}  +-  {mse_score_std}\n\n'
    scores += f'SSIM Score: {ssim_score}  +-  {ssim_score_std}'
    scores += "\n------------------------------Validation End------------------------------\n\n"
    print('------------------------------Validation End------------------------------')

    log_name = os.path.join(args.file, 'loss_log.txt')
    print(log_name)
    with open(log_name, "w") as log_file:
        log_file.write('%s\n' % scores)

    print('Finish!!!!!!!!!!!!')
