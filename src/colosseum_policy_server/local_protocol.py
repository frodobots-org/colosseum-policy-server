"""Binary Protobuf framing shared by the local control endpoints."""
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.message import DecodeError
from . import colosseum_pb2 as pb


def encode_control(value):
    # Dict conversion is internal only; the wire carries serialized Protobuf bytes.
    message = ParseDict(value, pb.LocalControl())
    return pb.RelayFrame(protocol_version=1, type=pb.LOCAL_CONTROL,
        session_id=message.run_id, payload=message.SerializeToString()).SerializeToString()


def decode_control(raw):
    if not isinstance(raw, bytes):
        raise ValueError('Binary Protobuf control frame required')
    try:
        frame = pb.RelayFrame.FromString(raw)
        if frame.protocol_version != 1 or frame.type != pb.LOCAL_CONTROL:
            raise ValueError('Invalid control frame version/type')
        message = pb.LocalControl.FromString(frame.payload)
        if not message.type or frame.session_id != message.run_id:
            raise ValueError('Invalid control message/session')
        return MessageToDict(message, preserving_proto_field_name=True)
    except DecodeError as exc:
        raise ValueError('Malformed Protobuf control frame') from exc
