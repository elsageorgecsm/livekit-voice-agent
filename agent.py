import logging
import time

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm, stt, tts, inference
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
from livekit.agents import function_tool, RunContext, ToolError
from livekit.agents import mcp
from livekit.agents import AgentTask
from livekit.agents.beta.workflows import TaskGroup
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


load_dotenv(".env")


@dataclass
class EmailResult:
    email_address: str


@dataclass
class AddressResult:
    address: str


class GetEmailTask(AgentTask[EmailResult]):
    def __init__(self) -> None:
        super().__init__(instructions="Collect the user's email address.")

    @function_tool()
    async def record_email(self, context: RunContext, email: str) -> None:
        """Record the user's email address"""
        self.complete(EmailResult(email_address=email))


class GetAddressTask(AgentTask[AddressResult]):
    def __init__(self) -> None:
        super().__init__(instructions="Collect the user's shipping address.")

    @function_tool()
    async def record_address(self, context: RunContext, address: str) -> None:
        """Record the user's shipping address"""
        self.complete(AddressResult(address=address))


class CheckoutAgent(Agent):
    async def on_enter(self) -> None:
        task_group = TaskGroup()

        task_group.add(
            lambda: GetEmailTask(), id="email", description="Collect email address"
        )

        task_group.add(
            lambda: GetAddressTask(),
            id="address",
            description="Collect shipping address",
        )

        results = await task_group

        email = results.task_results["email"].email_address
        address = results.task_results["address"].address

        await self.session.generate_reply(
            instructions=f"Confirm the order will be sent to {email} at {address}."
        )


class CollectConsent(AgentTask[bool]):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions="""
            Ask for recording consent and get a clear yes or no answer.
            Be polite and professional.
            """,
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="""
            Briefly introduce yourself, then ask for permission to record the call for quality assurance and training purposes.
            Make it clear that they can decline.
            """
        )

    @function_tool
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        self.complete(False)


class Manager(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                "You are a manager for a team of helpful voice AI assistants. "
                "Handle escalations professionally."
            ),
            tts="inworld/inworld-tts-1",
            chat_ctx=chat_ctx,
        )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly and professional voice AI assistant helping a customer place an order.

Speak naturally and keep responses short because this is a voice conversation.

Your goal is to collect the customer's contact and shipping details.

Information to collect:
1. Email address
2. Shipping address

Conversation Flow:
• First politely ask the user for their email address.
• After the user provides their email, call the tool `record_email`.
• Then ask for the user's shipping address.
• After the user provides the address, call the tool `record_address`.

Tool Usage Rules:
• When the user gives their email address, immediately call `record_email`.
• When the user gives their shipping address, immediately call `record_address`.

Conversation Guidelines:
• Be friendly and conversational.
• Ask for one piece of information at a time.
• Confirm politely if something is unclear.
• Do not mention internal tools or system instructions.

After collecting both the email and address, confirm the information with the user.
""",
        )

    @function_tool
    async def escalate_to_manager(self, context: RunContext):
        """Escalate the call to a manager on user request."""
        return Manager(chat_ctx=self.chat_ctx), "Escalating you to my manager now."

    async def on_enter(self) -> None:
        consent = await CollectConsent(chat_ctx=self.chat_ctx)

        if consent:
            await self.session.generate_reply(
                instructions="Thank them and offer your assistance."
            )
        else:
            await self.session.generate_reply(
                instructions="Let them know you understand and will proceed without recording consent."
            )


@function_tool
async def get_weather(
    context: RunContext,
    location: str,
) -> dict:
    """
    Look up current weather for a location.

    Args:
        location: City name or location to get weather for.
    """
    await context.agent_session.send_message(f"Looking up weather for {location}...")
    context.disallow_interruptions()

    async with httpx.AsyncClient() as client:
        # First, geocode the location to get coordinates
        geo_response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
        )

        geo_data = geo_response.json()

        if not geo_data.get("results"):
            raise ToolError(f"Could not find location: {location}")

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        place_name = geo_data["results"][0]["name"]

        # Get current weather for those coordinates
        weather_response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code",
                "temperature_unit": "fahrenheit",
            },
        )

        weather = weather_response.json()

        return {
            "location": place_name,
            "temperature_f": weather["current"]["temperature_2m"],
            "conditions": weather["current"]["weather_code"],
        }


server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):

    session = AgentSession(
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("assemblyai/universal-streaming:en"),
                inference.STT.from_model_string("deepgram/nova-3"),
            ]
        ),
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash"),
            ]
        ),
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string(
                    "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
                ),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        mcp_servers=[mcp.MCPServerHTTP(url="https://docs.livekit.io/mcp")],
    )
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        async def log_usage():
            summary = usage_collector.get_summary()
            logger.info("Usage summary: %s", summary)

        ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if (
            ev.new_state == "speaking"
            and last_eou_metrics
            and session.current_speech
            and last_eou_metrics.speech_id == session.current_speech.id
        ):
            delta = time.time() - last_eou_metrics.timestamp
            logger.info(f"Time to first audio frame:{delta:.3f} seconds")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
