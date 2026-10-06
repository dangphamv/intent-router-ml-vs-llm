"""Route messages: uv run scripts/route.py "Gói Pro có bao nhiêu seat?" "app bị lỗi 500"

(Example inputs are Vietnamese: "How many seats does the Pro plan have?", "the app has a 500 error".)
"""

import asyncio
import sys

from intent_router.router import IntentRouter


async def main(texts: list[str]):
    router = IntentRouter.load()
    for text in texts:
        r = await router.route(text)
        print(f"{r.label:<13} conf={r.confidence:.2f}  via={r.handled_by:<11} {text}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
