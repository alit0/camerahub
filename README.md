# CameraHub

Aplicación de visión por computadora para monitorear cámaras IP o locales con detección de personas y reconocimiento facial en tiempo real.

## Características

- Detección de personas mediante modelos preentrenados compatibles con OpenCV DNN (ej. MobileNet-SSD, YOLO).
- Reconocimiento facial basado en `face_recognition` y registro de rostros desde la cámara o imágenes cargadas.
- Distinción entre individuos conocidos y desconocidos con registro de eventos en SQLite.
- Interfaz gráfica en Tkinter que muestra el video, etiquetas superpuestas y log de eventos.
- Arquitectura modular orientada a distintos escenarios (hogar, oficina, gimnasio, etc.).

## Requisitos

- Python 3.8+
- Dependencias indicadas en `requirements.txt`
- Modelo de detección compatible (ejemplo recomendado: MobileNet-SSD `deploy.prototxt` y `res10_300x300_ssd_iter_140000_fp16.caffemodel`).

Instale las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Uso

1. Descargue los archivos del modelo de detección y ubíquelos en una carpeta accesible.
2. Inicie la aplicación con:

```bash
python main.py --camera 0 \
    --model-config /ruta/al/modelo.prototxt \
    --model-weights /ruta/al/modelo.caffemodel
```

Parámetros principales:

- `--camera`: índice de cámara local (ej. `0`) o URL RTSP/HTTP de cámara IP.
- `--width` / `--height`: dimensiones de captura.
- `--model-config`, `--model-weights`: rutas a los archivos del detector.
- `--class-names`: archivo opcional con nombres de clases.
- `--tolerance`: tolerancia para el reconocimiento facial (valores bajos son más estrictos).
- `--database`: ruta a la base de datos SQLite donde se almacenan eventos y rostros.

## Registro de rostros

- Desde la interfaz gráfica, pulse **Registrar rostro** para abrir el diálogo de registro.
- Asigne un nombre y capture una imagen desde la cámara activa o cargue una fotografía.
- El sistema extrae embeddings faciales y los guarda para futuras comparaciones.

## Escenarios de uso

- **Hogar:** identifique a los miembros de la familia y reciba alertas de intrusos.
- **Gimnasio:** reconozca socios autorizados y detecte accesos no registrados.
- **Oficina:** supervise entradas restringidas y mantenga un log de eventos.

## Registro de eventos

Los eventos se almacenan con marca de tiempo, etiqueta y estado (conocido/desconocido) en la base de datos indicada. Puede consultarse directamente con herramientas SQLite o extendiendo la aplicación.

## Estructura del proyecto

```
app/
├── camera.py          # Abstracción de cámara
├── config.py          # Configuración centralizada
├── detection.py       # Envoltura del detector de personas
├── gui.py             # Interfaz Tkinter
├── recognizer.py      # Pipeline de reconocimiento
├── registry.py        # Registro de rostros conocidos
└── storage.py         # Persistencia en SQLite
main.py                # Punto de entrada de la aplicación
requirements.txt       # Dependencias
```

## Notas

- Para mejorar el rendimiento en hardware limitado (p. ej. Raspberry Pi), reduzca la resolución de la cámara y utilice modelos ligeros.
- Puede integrar notificaciones externas conectando el registro de eventos con servicios de correo o mensajería.
