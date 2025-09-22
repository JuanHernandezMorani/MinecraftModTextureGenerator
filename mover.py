import os
import shutil

def mover_pngs():
    # Definir rutas de las carpetas
    carpeta_origen = 'toInput'
    carpeta_destino = 'input'
    
    # Asegurar que la carpeta destino existe
    os.makedirs(carpeta_destino, exist_ok=True)
    
    # Recorrer directorio y subdirectorios
    for raiz, _, archivos in os.walk(carpeta_origen):
        for archivo in archivos:
            if archivo.lower().endswith('.png'):
                ruta_completa = os.path.join(raiz, archivo)
                
                # Construir ruta destino
                destino = os.path.join(carpeta_destino, archivo)
                
                # Manejarchivos duplicados
                contador = 1
                nombre_base, extension = os.path.splitext(archivo)
                while os.path.exists(destino):
                    nuevo_nombre = f"{nombre_base}_{contador}{extension}"
                    destino = os.path.join(carpeta_destino, nuevo_nombre)
                    contador += 1
                
                # Mover el archivo
                shutil.move(ruta_completa, destino)
                print(f"Movido: {ruta_completa} -> {destino}")

if __name__ == "__main__":
    mover_pngs()