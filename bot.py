import os
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import numpy as np
import cv2
import logging
from diffusers import StableDiffusionPipeline
import io

# ← TOKEN ของคุณ
TOKEN = "8493364789:AAH84g-G8BNFlQy3We1yJaoM5Y1k6D-iSUg"

logging.basicConfig(level=logging.INFO)

# โหลด Stable Diffusion model (undress mode) - โหลดอัตโนมัติครั้งแรก
pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔥 Undress Bot โหดสุดพร้อมแล้ว!\n"
        "ส่งรูปผู้หญิงมาเลย ถอดให้เกลี้ยงภายใน 15 วิ\n"
        "/extreme → ถอดโคตรละเอียด\n"
        "/hentai → เปลี่ยนเป็นสไตล์เฮ็นไต"
    )

async def extreme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "extreme"
    await update.message.reply_text("⚡ โหมด Extreme เปิดแล้ว ถอดถึงเม็ดเหงื่อเลยครับ")

async def hentai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["mode"] = "hentai"
    await update.message.reply_text("🍓 โหมดเฮ็นไตเปิดแล้ว ส่งรูปมาเลย")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_chat_action("upload_photo")
    file = await update.message.photo[-1].get_file()
    await file.download_to_drive("input.jpg")

    mode = context.user_data.get("mode", "realistic")
    
    # อ่านรูป input
    init_image = Image.open("input.jpg")
    
    # Prompt สำหรับ undress แบบโหด (ปรับตาม mode)
    if mode == "extreme":
        prompt = "photorealistic nude woman, fully undressed, detailed skin, explicit, high resolution, realistic anatomy"
    elif mode == "hentai":
        prompt = "hentai style nude woman, anime, detailed, explicit, seductive pose"
    else:
        prompt = "photorealistic nude woman, undressed, natural lighting, high detail"
    
    negative_prompt = "clothed, blurry, deformed, ugly, extra limbs"
    
    # Generate ด้วย img2img (ใช้ init_image เป็น base)
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=0.75,  # ความแรงในการเปลี่ยน (0.75 = ถอดผ้าดี ๆ)
        guidance_scale=7.5,
        num_inference_steps=20,  # เร็วแต่คุณภาพดี
        negative_prompt=negative_prompt
    ).images[0]
    
    # บันทึกผล
    result.save("output_nude.jpg")
    
    await update.message.reply_photo(
        photo=open("output_nude.jpg", "rb"),
        caption="✅ เสร็จแล้วครับ เสียวไหมล่ะ 😈"
    )
    
    # ลบไฟล์ชั่วคราว
    os.remove("input.jpg")
    os.remove("output_nude.jpg")

app = Application.builder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("extreme", extreme))
app.add_handler(CommandHandler("hentai", hentai))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

print("🚀 บอทโหด ๆ เริ่มทำงานแล้ว...")
app.run_polling()