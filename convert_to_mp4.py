import subprocess

def create_video_from_images(input_folder, output_file, framerate=10, target_width=1920, target_height=1080):
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-pattern_type', 'glob',
        '-i', f'{input_folder}/*.jpg',  # Adjust the extension if necessary
        '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video '{output_file}' created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


create_video_from_images('/home/jose.viera/projects/cv802/neuralangelo/datasets/siamese_ds1/images_raw', 'einstein_downsampled.mp4')