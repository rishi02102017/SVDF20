"""
Training Configuration for SVDF-20 Dataset
"""
import argparse
import os

parser = argparse.ArgumentParser(description='SVDF-20 Training Configuration')

# Dataset paths
parser.add_argument('--database_path', type=str, 
                   default=os.getenv('SVDF_DATASET_PATH', './dataset'), 
                   help='Path to SVDF-20 dataset')
parser.add_argument('--dataset_name', type=str, default='SVDF-20', 
                   help='Dataset name')
parser.add_argument('--dataset_name_test', type=str, default='SVDF-20', 
                   help='Test dataset name')
parser.add_argument('--train_language', type=str, default='Multilingual', 
                   help='Training language')
parser.add_argument('--test_language', type=str, default='Multilingual', 
                   help='Test language')
parser.add_argument('--train_variant', type=str, default='Clean', 
                   help='Training variant')
parser.add_argument('--test_variant', type=str, default='Clean', 
                   help='Test variant')
parser.add_argument('--max_files', type=int, default=None, 
                   help='Maximum number of files to load (for testing)')

# Model parameters
parser.add_argument('--model_name', type=str, default='AASIST', 
                   help='Model name: AASIST, RawGAT_ST, RawNet2, SpecRNet, Whisper, SSLModel, Conformer, RawNetLite')
parser.add_argument('--xlsr_model_path', type=str, 
                   default=os.getenv('XLSR_MODEL_PATH', './src/models/xlsr2_300m.pt'),
                   help='Path to xlsr2_300m.pt model for SSLModel and Conformer')
parser.add_argument('--emb-size', type=int, default=144, 
                   help='Embedding size')
parser.add_argument('--heads', type=int, default=4, 
                   help='Number of attention heads')
parser.add_argument('--kernel_size', type=int, default=31, 
                   help='Kernel size for conv modules')
parser.add_argument('--num_encoders', type=int, default=4, 
                   help='Number of encoders')

# Training parameters
parser.add_argument('--batch_size', type=int, default=32, 
                   help='Batch size (will be reduced for Whisper model due to 30s audio)')
parser.add_argument('--num_epochs', type=int, default=25, 
                   help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-4, 
                   help='Learning rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                   help='Gradient accumulation for larger effective batch size')
parser.add_argument('--mixed_precision', type=bool, default=True,
                   help='Use mixed precision training (FP16) for speed')
parser.add_argument('--weight_decay', type=float, default=0.0001, 
                   help='Weight decay')
parser.add_argument('--pad_length', type=int, default=64600, 
                   help='Audio length to pad/truncate to (will be overridden by model-specific length)')
parser.add_argument('--fixed_length', default=True, 
                   type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                   help='Whether to use fixed length audio')

# Optimizer parameters
parser.add_argument('--optimizer', type=str, default='adam', 
                   help='Optimizer: adam, sgd')
parser.add_argument('--amsgrad', default=False, 
                   help='Use AMSGrad')
parser.add_argument('--base_lr', type=float, default=0.0001, 
                   help='Base learning rate')
parser.add_argument('--lr_min', type=float, default=0.000005, 
                   help='Minimum learning rate')
parser.add_argument('--betas', type=list, default=[0.9, 0.999], 
                   help='Betas for optimizer')
parser.add_argument('--scheduler', type=str, default='cosine', 
                   help='Scheduler type')
parser.add_argument('--use_scheduler', type=bool, default=False, 
                   help='Whether to use scheduler')

# Loss and evaluation
parser.add_argument('--loss', type=str, default='WCE', 
                   help='Loss function')
parser.add_argument('--weight_loss', type=list, default=[0.5, 0.5], 
                   help='Loss weights for bonafide/spoof')

# Output directories
parser.add_argument('--output_dir', type=str, 
                   default='/data-caffe/rishabh/SingFake_Project/IndicFake/models/svdf20_trained', 
                   help='Output directory for models')
parser.add_argument('--test_dir', type=str, 
                   default='/data-caffe/rishabh/SingFake_Project/IndicFake/results/svdf20_results', 
                   help='Output directory for results')
parser.add_argument('--metrics_path', type=str, default='metrics.csv', 
                   help='Metrics file path')

# Model loading/saving
parser.add_argument('--model_path', default='', 
                   help='Path to pretrained model checkpoint')
parser.add_argument('--eval', default=True, 
                   help='Whether to evaluate the model')
parser.add_argument('--train', default=True, 
                   type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                   help='Whether to train the model')

# Evaluation parameters
parser.add_argument('--n_mejores_loss', type=int, default=5, 
                   help='Save n-best models')
parser.add_argument('--average_model', default=True, 
                   type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                   help='Whether to average model weights')
parser.add_argument('--n_average_model', default=5, type=int, 
                   help='Number of models to average')
parser.add_argument('--eval_all_best', default=True, 
                   help='Evaluate all best models')

# Data augmentation (Rawboost)
parser.add_argument('--algo', type=int, default=5, 
                   help='Rawboost algorithm: 0=No aug, 1=LnL, 2=ISD, 3=SSI, 4=series(1+2+3), 5=series(1+2), 6=series(1+3), 7=series(2+3), 8=parallel(1,2)')
parser.add_argument('--nBands', type=int, default=5, 
                   help='Number of notch filters')
parser.add_argument('--minF', type=int, default=20, 
                   help='Minimum frequency [Hz]')
parser.add_argument('--maxF', type=int, default=8000, 
                   help='Maximum frequency [Hz]')
parser.add_argument('--minBW', type=int, default=100, 
                   help='Minimum bandwidth [Hz]')
parser.add_argument('--maxBW', type=int, default=1000, 
                   help='Maximum bandwidth [Hz]')
parser.add_argument('--minCoeff', type=int, default=10, 
                   help='Minimum filter coefficients')
parser.add_argument('--maxCoeff', type=int, default=100, 
                   help='Maximum filter coefficients')
parser.add_argument('--minG', type=int, default=0, 
                   help='Minimum gain factor')
parser.add_argument('--maxG', type=int, default=0, 
                   help='Maximum gain factor')
parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                   help='Minimum bias between linear/non-linear')
parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                   help='Maximum bias between linear/non-linear')
parser.add_argument('--N_f', type=int, default=5, 
                   help='Order of non-linearity')
parser.add_argument('--P', type=int, default=10, 
                   help='Maximum number of samples [%]')
parser.add_argument('--g_sd', type=int, default=2, 
                   help='Gain parameter')
parser.add_argument('--SNRmin', type=int, default=10, 
                   help='Minimum SNR')
parser.add_argument('--SNRmax', type=int, default=40, 
                   help='Maximum SNR')

# Other parameters
parser.add_argument('--seed', type=int, default=42, 
                   help='Random seed')
parser.add_argument('--comment', type=str, default=None, 
                   help='Comment for saved model')
parser.add_argument('--comment_eval', type=str, default=None, 
                   help='Comment for saved scores')
parser.add_argument('--eval_output', default='eval_scores.txt', 
                   help='Evaluation output file')

# Fine-tuning parameters
parser.add_argument('--FT_W2V', default=True, 
                   type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                   help='Whether to fine-tune W2V')

args = parser.parse_args()

# Create output directories if they don't exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.test_dir, exist_ok=True)
