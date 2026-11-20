# Ejecución con webcam de la laptop

Ejecuta el siguiente comando en la terminal:

    python target_trajectory_prediction.py --weights models\yolo_m_drone_v1.pt --source 0

# Ejecución con una camara conectada mediante IP.

1. Descargar la app **IP Webcam** para conectarnos mediante nuestro celular.
2. Conectar a la misma red nuestro celular y laptop (mediante un Access Point/WAP).
3. Iniciar la aplicación, eliminamos la contraseña y el usuario requeridos. Iniciamos el servidor. Comprobamos su conexión mediante un *ping*.
4. Ejecutamos el siguiente comando en la terminal:

    python target_trajectory_prediction.py --weights models\yolo_m_drone_v1.pt --source "http://x.x.x.x:8080/video"

    *Solo cambia la IP por la que tengas en el servidor*.

# Notas.
Para todos los casos anteriores, podemos añadir "--output salida.mp4" sin las comillas para generar una salida de video. 