
mkdir data
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/


cd /mnt

wget http://ai-challenger-scene.cn-bj.ufileos.com/ai_challenger_scene_train_20170904.zip
unzip ai_challenger_scene_train_20170904.zip

s3cmd get s3://ai.dataset/aic_data/val/ai_challenger_caption_validation_20170910.zip ./
unzip ai_challenger_caption_validation_20170910.zip

s3cmd get s3://ai.dataset/aic_data/test/ai_challenger_caption_test1_20170923.zip ./
unzip ai_challenger_caption_test1_20170923.zip



s3cmd get --recursive s3://ai.dataset/prepared_data/2017/10/06/ ./
