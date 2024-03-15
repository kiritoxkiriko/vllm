import argparse
import asyncio
import json
from contextlib import asynccontextmanager
import os
import importlib
import inspect

from prometheus_client import make_asgi_app
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest, ErrorResponse
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRA

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split('=')
            lora_list.append(LoRA(name, path))
        setattr(namespace, self.dest, lora_list)


# body logger
import time


def body_logger(request, raw_request: Request, start_time: float, resp=None):
    request_body = request.model_dump_json()
    process_time = time.time() - start_time
    request_id = raw_request.headers.get('X-NADP-RequestID')
    logger.info(
        f'receive request: id: {request_id}, body: {request_body}, resp: {resp}, time: {process_time}')


## add metrics
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    excluded_handlers=[".*admin.*", "/metrics", "/metrics-http"],
).instrument(app)


@app.on_event("startup")
async def _startup():
    # set ENABLE_METRICS to True to enable metrics
    instrumentator.expose(app, endpoint="/metrics-http")


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the header."
    )
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                             "specified, the model name will be the same as "
                             "the huggingface name.")
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help=
        "LoRA module configurations in the format name=path. Multiple modules can be specified."
    )
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                             "or the template in single-line form "
                             "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                             "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
             "We accept multiple --middleware arguments. "
             "The value should be an import path. "
             "If a function is provided, vLLM will add it to the server using @app.middleware('http'). "
             "If a class is provided, vLLM will add it to the server using app.add_middleware(). "
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())

########################## NIO hot fix ##########################
import numpy as np
import json
from vllm.entrypoints.openai.faiss_ip_index import FaissSentenceIndexer, knowledge_tagging
from tqdm.auto import tqdm
import requests
embedding_api_url = "http://172.26.181.124:8089"
reranker_api_url = "http://172.26.181.66:8089"
proxies = {
    'http': 'socks5://10.161.35.14:32101',
    'https': 'socks5://10.161.35.14:32101'
}
def embedding_get(sentences, api_url=embedding_api_url,proxies=proxies,batch_size=16):
    # auto batching
    for i in tqdm(range(0, len(sentences), batch_size)):
        response = requests.post(api_url + '/predictions/SBERT', data={'data': json.dumps({'queries': sentences[i:i+batch_size]})},proxies=proxies)
        if response.status_code == 200:
            vectors = response.json()
        else:
            return response.text
        if i == 0:
            all_vectors = vectors
        else:
            all_vectors.extend(vectors)
    return all_vectors

def reranking_reverse_get(query, candidates, api_url=reranker_api_url,proxies=proxies,batch_size=8):
    pairs = [[query, i] for i in candidates]
    # auto batching
    for i in tqdm(range(0, len(pairs), batch_size)):
        response = requests.post(api_url + '/predictions/SBERT', data={'data': json.dumps({'queries': pairs[i:i+batch_size]})},proxies=proxies)
        if response.status_code == 200:
            vectors = response.json()
        else:
            return response.text
        if i == 0:
            all_vectors = vectors
        else:
            all_vectors.extend(vectors)
    if isinstance(all_vectors, list):
        return sorted(zip(candidates, all_vectors), key=lambda x: x[1], reverse=True)

def retrival(query,sentence_indexer, embedding_api_url=embedding_api_url, reranker_api_url=reranker_api_url, proxies=proxies, topk=10, rerank=True, using_internal_keyword_system=False):
    query_embedding = embedding_get([query], embedding_api_url, proxies)
    query_embedding = np.array(query_embedding).astype('float32')
    result = sentence_indexer.search(query_embedding, k=topk)
    # 获取句子原句
    candidates = result[2][0]
    if rerank:
        result = reranking_reverse_get(query, candidates, reranker_api_url ,proxies)
        return [i[0] for i in result]
    else:
        return candidates

def build_faiss_indexer(_sentences, embedding_api_url=embedding_api_url, proxies=proxies, dim=1024):
    _embeddings = np.array(embedding_get(_sentences, embedding_api_url, proxies)).astype('float32')
    # _keyword_set_list = [knowledge_tagging(i) for i in _sentences]
    _keyword_set_list = []
    sentence_indexer = FaissSentenceIndexer(_embeddings, _sentences, _keyword_set_list, d=dim,silence=True)
    return sentence_indexer, _sentences, _embeddings


def add_doc2_request(request):
    for m in request.messages:
        if m['role'] == 'user':
            if '@@selected_knowledges@@' in m['content']:
                user_query = m['content'].split('#用户提问：')[-1].strip()
                selected_knowledges = retrival(user_query, sentence_indexer, embedding_api_url, reranker_api_url, proxies,topk=10, rerank=False, using_internal_keyword_system=False)
                selected_knowledges = json.dumps(selected_knowledges, ensure_ascii=False,indent=4)
                m['content'] = m['content'].replace('@@selected_knowledges@@',selected_knowledges)
                break
        else:
            continue
    return request

##############################123#####################################

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    try:
        request = add_doc2_request(request)
        print(request.messages[-1])
    except Exception as e:
        print(e)
        print(e)
        print(e)

    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        body_logger(request, raw_request, time.time(), generator.model_dump())
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model, args.lora_modules)

    ######################### NIO hot fix #########################
    try:
        with open('/ur-hl/wenrui.zhou/baas_sentences.json', 'r') as f:
            sentences = json.load(f)
            sentence_indexer, _sentences, _embeddings = build_faiss_indexer(sentences, dim=768)
    except Exception as e:
        print(e)
        print(e)
        print(e)



    ######################### NIO hot fix #########################

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
