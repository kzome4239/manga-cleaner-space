import os
import shutil
import subprocess
import sys
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

# استنسخ مستودع comic-text-detector إذا لم يكن موجوداً
if not os.path.exists("comic-text-detector"):
    subprocess.run(["git", "clone", "https://github.com/dmMaze/comic-text-detector.git"])

sys.path.append("comic-text-detector")

# تنزيل النموذج المضمون من HuggingFace Hub
os.makedirs("comic-text-detector/data", exist_ok=True)
model_path = "comic-text-detector/data/comictextdetector.pt"

if not os.path.exists(model_path):
    print("Downloading model from HuggingFace Hub...")
    downloaded_file = hf_hub_download(
        repo_id="ogkalu/comic-text-detector", 
        filename="comictextdetector.pt"
    )
    shutil.copy(downloaded_file, model_path)

@spaces.GPU
def clean_manga_images(files):
    if not files:
        return None, "رجاءً قم برفع الصور أولاً."

    input_dir = "input_chapter"
    output_dir = "cleaned_chapter"

    if os.path.exists(input_dir): shutil.rmtree(input_dir)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for file in files:
        filename = os.path.basename(file.name)
        shutil.copy(file.name, os.path.join(input_dir, filename))

    cmd = [
        sys.executable, "comic-text-detector/inference.py",
        "--img_dir", input_dir,
        "--save_dir", output_dir,
        "--model_path", model_path
    ]
    subprocess.run(cmd)

    cleaned_files = os.listdir(output_dir)
    if not cleaned_files:
        return None, "لم يتم تبييض الصور، المجلد الناتج فارغ."

    zip_path = "cleaned_manga"
    shutil.make_archive(zip_path, "zip", output_dir)

    return "cleaned_manga.zip", f"تم تنظيف الفصل بنجاح! عدد الصور المعالجة: {len(cleaned_files)}"

with gr.Blocks(title="منظف المانجا التلقائي") as demo:
    gr.Markdown("# 🎨 موقع تنظيف المانجا والكوميكس بالذكاء الاصطناعي")
    gr.Markdown("ارفع صور الفصل وسيقوم الموقع بتنظيف فقاعات النصوص تلقائياً بدون استهلاك موارد جهازك.")

    file_input = gr.File(label="ارفع صور الفصل هنا", file_count="multiple", file_types=["image"])
    clean_btn = gr.Button("بدء التنظيف التلقائي 🚀", variant="primary")

    file_output = gr.File(label="تحميل الفصل المنظف (ZIP)")
    status_output = gr.Textbox(label="الحالة")

    clean_btn.click(
        fn=clean_manga_images,
        inputs=[file_input],
        outputs=[file_output, status_output]
    )

demo.launch()
