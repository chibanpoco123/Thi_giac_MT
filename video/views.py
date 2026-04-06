import os
import time
from django.shortcuts import render
from django.conf import settings
from .stabilizer import stabilize_video, create_comparison_video, measure_shakiness


def home(request):
    output_video = None
    input_video = None
    compare_video = None
    score = None

    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        input_dir = os.path.join(settings.MEDIA_ROOT, 'input')
        output_dir = os.path.join(settings.MEDIA_ROOT, 'output')

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = int(time.time())

        input_filename = f"input_{timestamp}.mp4"
        output_filename = f"output_{timestamp}.mp4"

        input_path = os.path.join(input_dir, input_filename)
        output_path = os.path.join(output_dir, output_filename)

        # save
        with open(input_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # ===== MEASURE BEFORE =====
        before_mean, before_std = measure_shakiness(input_path)

        # ===== STABILIZE =====
        result = stabilize_video(input_path, output_path)

        if result and os.path.exists(result):

            # ===== MEASURE AFTER =====
            after_mean, after_std = measure_shakiness(result)

            if before_std and after_std:
                score = round(((before_std - after_std) / before_std) * 100, 2)

            # ===== COMPARE =====
            compare_filename = f"compare_{timestamp}.mp4"
            compare_path = os.path.join(output_dir, compare_filename)

            compare_result = create_comparison_video(
                input_path,
                result,
                compare_path
            )

            input_video = f'/media/input/{input_filename}'
            output_video = f'/media/output/{os.path.basename(result)}'

            if compare_result:
                compare_video = f'/media/output/{os.path.basename(compare_result)}'

    return render(request, 'home.html', {
        'output_video': output_video,
        'input_video': input_video,
        'compare_video': compare_video,
        'score': score
    })