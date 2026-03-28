import numpy as np


def evaluate(gt_path, pred_path):
    """
    Evaluate model predictions by computing the MAE of heading_num and the MAE of range_num.
    GT format: each line is "heading_num, range_num" (comma-separated)
    Pred format: each line is "heading_num range_num" (space-separated)
    """
    heading_errors = []
    range_errors = []

    with open(gt_path, 'r') as f_gt, open(pred_path, 'r') as f_pred:
        for line_gt, line_pred in zip(f_gt, f_pred):
            line_gt = line_gt.strip()
            line_pred = line_pred.strip()
            if not line_gt or not line_pred:
                continue

            # Parse GT (comma-separated)
            parts_gt = line_gt.split(',')
            gt_heading = float(parts_gt[0].strip())
            gt_range = float(parts_gt[1].strip())

            # Parse Pred (space-separated)
            parts_pred = line_pred.split()
            pred_heading = float(parts_pred[0].strip())
            pred_range = float(parts_pred[1].strip())

            # Heading error: wrapped angle difference, mapped to [-180, 180]
            delta_heading = pred_heading - gt_heading
            delta_heading = (delta_heading + 180.0) % 360.0 - 180.0
            heading_errors.append(abs(delta_heading))

            # Range error: absolute difference
            range_errors.append(abs(pred_range - gt_range))

    heading_errors = np.array(heading_errors)
    range_errors = np.array(range_errors)

    heading_mae = heading_errors.mean()
    range_mae = range_errors.mean()

    print(f"Total samples: {len(heading_errors)}")
    print(f"Heading MAE: {heading_mae:.4f} degrees")
    print(f"Range   MAE: {range_mae:.4f}")


if __name__ == '__main__':
    gt_path = './test_gt.txt'
    pred_path = './test_predict_output.txt'
    evaluate(gt_path, pred_path)