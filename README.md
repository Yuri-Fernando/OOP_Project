# VisionGuard — Real-Time Object Detection & Event Trigger System

### Computer Vision · YOLOv8 · OpenCV · Event-Driven Vision · Object Detection · Python

## Status

🟢 **Concluído — Projeto de portfólio / Computer Vision**

O **VisionGuard** é um sistema de visão computacional em tempo real desenvolvido para detectar objetos e acionar ações apenas quando ocorre um **evento relevante**, evitando o processamento lógico e armazenamento de detecções redundantes.

O projeto utiliza **YOLOv8**, OpenCV e uma arquitetura modular orientada a objetos para transformar a detecção convencional em um pipeline de visão baseado em eventos.

---

## Sobre o Projeto

Em vez de processar e armazenar continuamente todas as detecções realizadas pela câmera, o sistema mantém o estado observado e gera uma saída somente quando identifica um **novo objeto-alvo**.

Fluxo principal:

```text
Webcam
   ↓
Frame
   ↓
YOLOv8
   ↓
Detecções
   ↓
Filtro de Classes
   ↓
Comparação com Estado Anterior
   ↓
Novo Objeto?
   ├── Não → Continua monitorando
   └── Sim → Evento
                    ↓
              Captura de imagem
                    ↓
              Visualização / Output
```

Essa abordagem reduz saídas redundantes e cria uma base para sistemas orientados a eventos.

---

# Objetivo

Demonstrar, de forma prática, como evoluir de um pipeline tradicional de detecção de objetos para uma arquitetura capaz de:

- Detectar objetos em tempo real;
- Filtrar classes relevantes;
- Identificar novos eventos;
- Evitar capturas redundantes;
- Organizar o código em módulos independentes;
- Separar inferência, regras de negócio e visualização;
- Criar uma base extensível para alertas, APIs e automações.

---

# Funcionalidades

- Detecção de objetos em tempo real com **YOLOv8**;
- Captura de vídeo por webcam;
- Arquitetura modular baseada em OOP;
- Processamento orientado a eventos;
- Identificação de novos objetos;
- Captura automática de imagens;
- Filtro de classes;
- Ignorar classes irrelevantes, como `person`;
- Visualização lado a lado de frame original e processado;
- Suporte a Jupyter Notebook;
- Armazenamento automático das detecções relevantes.

---

# Arquitetura

O projeto foi organizado com separação clara de responsabilidades.

```text
                    ┌──────────────────┐
                    │     Webcam       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Pipeline     │
                    │  Orchestration    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Detector     │
                    │     YOLOv8       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Processor     │
                    │ Filtering / Rules│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Event Detection  │
                    │ State Comparison  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Visualizer    │
                    │ Boxes / Labels   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      Output      │
                    │ Capture / Images │
                    └──────────────────┘
```

---

# Componentes

## Detector

Arquivo:

```text
src/core/detector.py
```

Responsável por:

- Carregar o modelo YOLO;
- Executar inferência;
- Processar os frames;
- Retornar as detecções.

---

## Processor

Arquivo:

```text
src/core/processor.py
```

Responsável por:

- Aplicar regras de negócio;
- Filtrar classes;
- Ignorar objetos irrelevantes;
- Selecionar objetos-alvo;
- Comparar o estado atual com o anterior;
- Determinar quando um evento deve ser disparado.

---

## Visualizer

Arquivo:

```text
src/core/visualizer.py
```

Responsável por:

- Desenhar bounding boxes;
- Exibir labels;
- Preparar a visualização dos frames;
- Comparar frame original e processado.

---

## Pipeline

Arquivo:

```text
src/core/pipeline.py
```

Responsável por integrar os demais componentes em um fluxo único.

```text
Capture
   ↓
Detect
   ↓
Process
   ↓
Evaluate Event
   ↓
Visualize
   ↓
Save Output
```

---

# Event-Driven Detection

O principal diferencial do VisionGuard está na lógica orientada a eventos.

Em um sistema convencional:

```text
Frame 1 → Detecção → Salvar
Frame 2 → Detecção → Salvar
Frame 3 → Detecção → Salvar
Frame 4 → Detecção → Salvar
```

Isso pode gerar grande quantidade de informações redundantes.

No VisionGuard:

```text
Frame 1 → Objeto detectado
Frame 2 → Mesmo objeto → Ignorar
Frame 3 → Mesmo objeto → Ignorar
Frame 4 → Novo objeto → EVENTO
```

Resultado:

```text
Novo objeto
   ↓
Evento
   ↓
Captura
   ↓
Output
```

Essa lógica cria uma base para aplicações de monitoramento em que o evento, e não cada frame, é o principal elemento de interesse.

---

# Exemplo de Uso

A implementação atual considera objetos como:

```text
cell phone
```

como alvo de detecção, enquanto classes irrelevantes podem ser ignoradas.

Exemplo:

```text
New detection: ['cell phone']
        ↓
outputs/detect_1700000000.jpg
```

O arquivo é salvo somente quando um novo objeto relevante é identificado.

---

# Casos de Uso

A arquitetura pode servir como base para:

### Surveillance

Captura apenas quando novos objetos ou eventos relevantes surgem.

### Industrial Monitoring

Monitoramento orientado a eventos em ambientes industriais.

### Robotics

Percepção baseada em mudanças significativas no ambiente.

### Smart Automation

Acionamento de workflows a partir de eventos visuais.

### Event-Driven Vision

Integração de visão computacional com:

- APIs;
- Webhooks;
- Alertas;
- Mensageria;
- Sistemas de automação.

---

# Tecnologias

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python |
| Computer Vision | OpenCV |
| Object Detection | Ultralytics YOLOv8 |
| Visualização | Matplotlib |
| Experimentação | Jupyter Notebook |
| Arquitetura | Object-Oriented Programming |
| Paradigma | Event-Driven Processing |

---

# Estrutura do Projeto

```text
vision_project/
│
├── src/
│   ├── core/
│   │   ├── detector.py
│   │   ├── processor.py
│   │   ├── visualizer.py
│   │   └── pipeline.py
│   │
│   └── utils/
│       └── helpers.py
│
├── notebooks/
│   └── demo.ipynb
│
├── outputs/
│   ├── detect_1700000000.jpg
│   └── detect_1700000005.jpg
│
├── requirements.txt
└── README.md
```

---

# Como Executar

## Requisitos

- Python;
- Webcam;
- Jupyter Notebook ou Jupyter Lab;
- Dependências do projeto.

## Instalar dependências

```bash
pip install ultralytics opencv-python matplotlib
```

## Executar

Abra:

```text
notebooks/demo.ipynb
```

Execute todas as células para iniciar o sistema de detecção.

---

# Controles

Durante a execução pelo Jupyter:

- Utilize o botão de parada do notebook para interromper o processamento;
- O sistema deve liberar a câmera após a interrupção.

---

# Outputs

As imagens associadas a novos eventos são armazenadas automaticamente em:

```text
outputs/
├── detect_1700000000.jpg
├── detect_1700000005.jpg
└── ...
```

O sistema evita salvar imagens quando a detecção corresponde ao mesmo estado observado anteriormente.

---

# O que este projeto demonstra

- Object Detection;
- YOLOv8;
- Computer Vision;
- OpenCV;
- Processamento em tempo real;
- Arquitetura orientada a objetos;
- Separação de responsabilidades;
- Event-Driven Architecture;
- State Comparison;
- Filtering Logic;
- Automatic Event Triggering;
- Processamento de imagens;
- Integração entre inferência e regras de negócio;
- Construção de pipelines modulares de visão computacional.

---

# Limitações

- A performance em Jupyter pode não representar a velocidade máxima possível devido às limitações de renderização do notebook;
- Para processamento contínuo com maior desempenho, uma execução como script Python pode ser mais adequada;
- O modelo YOLO utilizado é baseado no dataset COCO;
- Algumas classes podem não ser identificadas com precisão suficiente para aplicações específicas;
- Smartwatches, por exemplo, podem ser classificados como `cell phone`;
- O sistema atual utiliza comparação de estado simples e não implementa tracking persistente entre objetos.

---

# Melhorias Futuras

- Object Tracking com DeepSORT;
- Tracking multiobjeto;
- API com FastAPI;
- Alertas via Telegram;
- Integração com Webhooks;
- Logging estruturado;
- Persistência dos eventos em banco de dados;
- Dashboard de eventos;
- Modelos customizados para objetos específicos;
- Edge deployment;
- Execução em Jetson Nano;
- Execução em Raspberry Pi;
- Integração com sistemas de automação;
- Pipeline de analytics sobre os eventos detectados.

---

# Status Final

🟢 **Concluído**

A versão atual possui:

- ✅ YOLOv8;
- ✅ OpenCV;
- ✅ Detecção em tempo real;
- ✅ Arquitetura OOP;
- ✅ Filtro de classes;
- ✅ Event-based triggering;
- ✅ Comparação com estado anterior;
- ✅ Captura automática de novas detecções;
- ✅ Visualização;
- ✅ Notebook de demonstração;
- ✅ Armazenamento de outputs;
- ✅ Estrutura modular.

O projeto permanece como uma base para evolução de sistemas de visão computacional **orientados a eventos**, podendo ser expandido para monitoramento, robótica, automação e sistemas inteligentes.

---

# Licença

Consulte a licença definida no repositório.

---

---

# Versão 2.0 — Extensão de Arquitetura

A V1 para no YOLO: detectar, filtrar, desenhar. A V2 mantém esse pipeline intacto e adiciona
um segundo estágio de CNN sobre ele — um classificador **ResNet** rodando sobre o recorte de
cada objeto detectado.

```text
VisionGuard
     │
     ├── YOLOv8 → Object Detection
     │
     └── ResNet → Image Classification / Feature Extraction
```

## O que foi adicionado

- `src/core/classifier_v2.py` — **ResNetClassifier**: encapsula uma ResNet pré-treinada do
  `torchvision` (`resnet18` por padrão, com suporte a `resnet34`/`resnet50`) e expõe:
  - `classify(crop)` → top-k rótulos ImageNet + confiança para o recorte de um objeto detectado;
  - `extract_features(crop)` → embedding da penúltima camada (feature extraction), disponível
    para casos futuros de busca por similaridade.
- `src/core/pipeline_v2.py` — **VisionPipelineV2**: estende `VisionPipeline` (V1 permanece
  intacta) e, após o fluxo existente de detectar → filtrar → desenhar, roda o classificador
  ResNet sobre o recorte de cada detecção-alvo, adicionando o campo `resnet_classification`.
- `demo_v2.ipynb` — mesmo fluxo de webcam do `demo.ipynb`, agora exibindo a classificação
  ResNet ao lado do rótulo bruto do YOLO/COCO a cada novo evento.
- `requirements_v2.txt` — adiciona `torch` e `torchvision` às dependências da V1.

## Por quê

A detecção de objetos (YOLO) responde *"onde está, e qual classe do COCO é?"*. Adicionar um
estágio de classificação com rede residual sobre cada recorte responde uma pergunta mais
específica — uma categoria ImageNet mais granular, além de um embedding que pode futuramente
sustentar busca por similaridade entre objetos detectados — sem alterar nem impactar o
caminho de detecção já existente (a V2 é estritamente aditiva: `Detector`, `DetectionProcessor`,
`Visualizer` e `VisionPipeline` da V1 continuam inalterados).

## V1 vs V2

| | V1 | V2 |
|---|---|---|
| Estágios | Apenas detecção YOLOv8 | Detecção YOLOv8 + classificação ResNet |
| Saída por detecção | Rótulo COCO + confiança | Rótulo COCO + top-k rótulos ImageNet (ResNet) |
| Feature extraction | — | Embedding ResNet via `extract_features` |

Execute `demo_v2.ipynb` da mesma forma que `demo.ipynb` (requer webcam); instale
`requirements_v2.txt` antes.

---

# Autor

**Yuri Fernando Dubbern**

AI/ML Engineer · Computer Vision · Machine Learning · Intelligent Automation

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)
