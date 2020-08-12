import dpkt
import collections

from datetime import datetime as dt


def parse_pkts_per_second(filename):
    """ Parses pcap file, counting packets per second """

    packets_per_second = {}
    counter, start, end = 0, 0, 0
    with open(filename, 'rb') as pcap_file:
        for ts, pkt in dpkt.pcap.Reader(pcap_file):
            if counter == 0:
                start = int(ts)
            end = int(ts)
            counter += 1

            # timestamp = dt.utcfromtimestamp(ts)
            # key = '{}{}{}'.format(timestamp.hour, timestamp.minute, timestamp.second)
            key = int(ts)
            if key not in packets_per_second:
                packets_per_second[key] = 0
            packets_per_second[key] += 1

    # print('start, end, diff, counter', start, end, end - start, counter, len(packets_per_second.items()))
    # cover seconds with no packets (if any)
    count = 0
    for sec in range(start, end + 1):
        if sec not in packets_per_second:
            count += 1
            packets_per_second[sec] = 0

    # print('count', count, len(packets_per_second.items()))
    ordered = collections.OrderedDict(sorted(packets_per_second.items()))
    return list(ordered.values())


def parse_pkts_per_millisecond(filename):
    """ Parses pcap file, counting packets per millisecond """

    packets_per_millisecond = {}
    counter, start, end = 0, 0, 0
    with open(filename, 'rb') as pcap_file:
        for ts, pkt in dpkt.pcap.Reader(pcap_file):
            if counter == 0:
                start = int(ts * 1000)
            end = int(ts * 1000)
            counter += 1

            # timestamp = dt.utcfromtimestamp(ts)
            # key = '{}'.format(timestamp.hour, timestamp.minute, timestamp.second, timestamp.microsecond / 1000)
            key = int(ts * 1000)
            if key not in packets_per_millisecond:
                packets_per_millisecond[key] = 0
            packets_per_millisecond[key] += 1

    # print('start, end, diff, counter', start, end, end - start, counter, len(packets_per_millisecond.items()))

    # cover milliseconds with no packets (if any)
    count = 0
    for milli in range(start, end + 1):
        if milli not in packets_per_millisecond:
            count += 1
            packets_per_millisecond[milli] = 0

    # print('count', count, len(packets_per_millisecond.items()))
    ordered = collections.OrderedDict(sorted(packets_per_millisecond.items()))
    return list(ordered.values())


def parse_pkts_per_second_per_flow(filename):
    """ Parses pcap file, dividing by flow """

    packets_per_second = {}
    with open(filename, 'rb') as pcap_file:
        for ts, pkt in dpkt.pcap.Reader(pcap_file):
            timestamp = dt.utcfromtimestamp(ts)
            key = '{}{}{}'.format(timestamp.hour, timestamp.minute, timestamp.second)
            if key not in packets_per_second:
                packets_per_second[key] = 0
            packets_per_second[key] += 1

    return list(packets_per_second.values())
