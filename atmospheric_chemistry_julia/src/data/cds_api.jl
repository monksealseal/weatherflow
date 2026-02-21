# ---------------------------------------------------------------------------
# CDS API client for downloading ERA5 data from Copernicus Climate Data Store
# ---------------------------------------------------------------------------

"""
    CDSClient(; url, key)

Minimal client for the Copernicus Climate Data Store (CDS) API.
Reads credentials from `~/.cdsapirc` if not provided.

The `.cdsapirc` file should contain:
```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```
"""
mutable struct CDSClient
    url :: String
    key :: String
end

function CDSClient(; url::String="", key::String="")
    if isempty(url) || isempty(key)
        rc_path = joinpath(homedir(), ".cdsapirc")
        if isfile(rc_path)
            for line in readlines(rc_path)
                line = strip(line)
                if startswith(line, "url:")
                    url = strip(line[5:end])
                elseif startswith(line, "key:")
                    key = strip(line[5:end])
                end
            end
        end
    end

    if isempty(url)
        url = "https://cds.climate.copernicus.eu/api"
    end

    if isempty(key)
        @warn "No CDS API key found. Set it in ~/.cdsapirc or pass key= to CDSClient. " *
              "Register at https://cds.climate.copernicus.eu/"
    end

    return CDSClient(url, key)
end

"""
    submit_request(client, dataset, request; output_path)

Submit an asynchronous retrieval request to CDS and poll until complete,
then download the result to `output_path`.
"""
function submit_request(client::CDSClient, dataset::String,
                        request::Dict; output_path::String)
    if isfile(output_path)
        @info "File already exists, skipping download: $output_path"
        return output_path
    end

    mkpath(dirname(output_path))

    headers = [
        "Content-Type" => "application/json",
    ]

    # Encode credentials
    auth_header = "Basic " * base64encode(client.key)
    push!(headers, "Authorization" => auth_header)

    url = "$(client.url)/resources/$(dataset)"

    @info "Submitting CDS request for $dataset..."
    body = JSON3.write(request)

    resp = HTTP.post(url, headers, body; status_exception=false)

    if resp.status == 202 || resp.status == 200
        result = JSON3.read(String(resp.body))

        if haskey(result, :request_id)
            request_id = result.request_id
            @info "Request submitted: $request_id. Polling for completion..."
            return _poll_and_download(client, request_id, output_path, headers)
        elseif haskey(result, :location)
            # Direct download available
            return _download_file(result.location, output_path, headers)
        end
    end

    @error "CDS request failed with status $(resp.status)" response=String(resp.body)
    error("CDS API request failed")
end

function _poll_and_download(client::CDSClient, request_id::String,
                            output_path::String, headers)
    max_wait = 3600  # 1 hour
    elapsed = 0
    interval = 10

    while elapsed < max_wait
        sleep(interval)
        elapsed += interval

        url = "$(client.url)/tasks/$(request_id)"
        resp = HTTP.get(url, headers; status_exception=false)
        result = JSON3.read(String(resp.body))

        state = get(result, :state, "unknown")
        if state == "completed"
            download_url = result.location
            return _download_file(download_url, output_path, headers)
        elseif state == "failed"
            error("CDS request failed: $(get(result, :error, "unknown error"))")
        else
            @info "CDS request state: $state ($(elapsed)s elapsed)"
        end

        # Exponential backoff, capped at 60s
        interval = min(interval * 2, 60)
    end

    error("CDS request timed out after $(max_wait)s")
end

function _download_file(url::String, output_path::String, headers)
    @info "Downloading to $output_path..."
    resp = HTTP.get(url, headers)
    write(output_path, resp.body)
    @info "Download complete: $(filesize(output_path) / 1e6) MB"
    return output_path
end

"""Helper to base64-encode a string (for Basic auth)."""
function base64encode(s::String)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    bytes = Vector{UInt8}(s)
    result = IOBuffer()
    i = 1
    while i <= length(bytes)
        b1 = bytes[i]
        b2 = i + 1 <= length(bytes) ? bytes[i+1] : UInt8(0)
        b3 = i + 2 <= length(bytes) ? bytes[i+2] : UInt8(0)

        write(result, chars[(b1 >> 2) + 1])
        write(result, chars[((b1 & 0x03) << 4 | b2 >> 4) + 1])

        if i + 1 <= length(bytes)
            write(result, chars[((b2 & 0x0f) << 2 | b3 >> 6) + 1])
        else
            write(result, '=')
        end

        if i + 2 <= length(bytes)
            write(result, chars[(b3 & 0x3f) + 1])
        else
            write(result, '=')
        end

        i += 3
    end
    return String(take!(result))
end
