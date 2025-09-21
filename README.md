# Minecraft Texture Variant Generator 🎨⚡

Herramienta en **Python** para generar automáticamente variantes de texturas para mobs y entidades de Minecraft.  
Procesa imágenes base (`input/`) y genera versiones **zombie, frozen, corrupted, toxic, oceanic, nether y celestial**, incluyendo:

- Textura difusa (`.png`)
- Mapa normal (`_n.png`) generado con Sobel
- Mapa emissive (`_e.png`) opcional según máscaras

Los resultados se organizan en `output/<variante>/`, listos para integrarse en tu mod.

---

## 🚀 Características
- **Batch automático**: procesa todas las texturas de `input/` en una sola ejecución.
- **Tintes HSV** configurables por variante.
- **Overlays** opcionales por variante (ej. grietas de hielo, venas, aureolas).
- **Máscaras emissive** opcionales (ojos brillantes, cristales, aura).
- **Normal maps básicos** para relieve en shaders.
- **Robusto y escalable**: fácil de extender a más variantes.

---

## 📂 Estructura de Carpetas
```plaintext
texture_generator/
├── input/                 # Texturas base (ej: zombie.png, cow.png)
├── overlays/              # Overlays opcionales (ej: frozen.png)
├── masks/                 # Máscaras opcionales para emissive (ej: corrupted.png)
├── output/
│   ├── zombie/
│   ├── frozen/
│   ├── corrupted/
│   ├── toxic/
│   ├── oceanic/
│   ├── nether/
│   └── celestial/
└── generate.py            # Script principal
