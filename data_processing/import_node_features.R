#install.packages(c("taxize","dplyr","tidyr","jsonlite"))
library(taxize)
library(dplyr)
library(tidyr)
library(jsonlite)

# As taxonomic databases may return empty results or multiple results
#   for a single species/node name, we take the first (best) result and NAs
first_or_na <- function(x) if (length(x) && !is.na(x[[1]])) x[[1]] else NA

# Infix OR operator
`%||%` <- function(a, b)
  if (is.null(a) || !length(a) || is.na(a[1]) || !nzchar(a[1])) b else a

# As node/species names contained within the food webs from web of life are not
#   necessarily in binomial nomenclature, we need to resolve them to their
#   canonical names to be able to query the taxonomic databases
resolve_name <- function(name, verbose = FALSE) {
    nm <- trimws(name)
    if (!nzchar(nm)) return(NA_character_)

    # Attempt to use gnr_resolve to first as it can handle misspellings.
    #   Though gnr_resolve is deprecated, the latest alternative (gna_verifier)
    #   cannot handle typos, hence why gnr_resolve is used instead
    gn <- tryCatch(
        gnr_resolve(
            sci = nm,
            canonical = TRUE,
            best_match_only = TRUE,
            fields = "all"
        ),
        error = function(e) NULL
    )
    # If matched_name2 is available, we use this as matched_name contains
    #   additional metadata in the returned string
    if (!is.null(gn) && nrow(gn)) {
        canon <- gn$matched_name2[1] %||% gn$matched_name[1]
        if (verbose) cat("     gnr_resolve →", canon, "\n")
        return(canon)
    }

    # If gnr_resolve fails, try comm2sci as its more lenient with common names
    cs <- tryCatch(
        comm2sci(nm, simplify = TRUE), 
        error = function(e) NULL
    )
    if (!is.null(cs) && length(cs) && !is.na(cs[[1]][1]) && nzchar(cs[[1]][1])) {
        canon <- cs[[1]][1]
        if (verbose) cat("     comm2sci   →", canon, "\n")
        return(canon)
    }

    # If both fail, attempt search with original node/species name
    nm
}

# Different taxonomic databases have different column names for the same
#   information, so we need to normalise them to a common format
#   before we can merge them together
clean_classification_df <- function(df) {
  # Ignore empty dataframes
  if (is.null(df) || !is.data.frame(df) || nrow(df) == 0)
    return(NULL)

  # Normalise column names across databases
  nms <- tolower(names(df))
  names(df)[nms %in% c("taxonrank", "taxonrankname")] <- "rank"
  names(df)[nms %in% c("scientificname", "taxonname")] <- "name"
  names(df)[nms == "key" & !("id" %in% names(df))] <- "id"

  # If fields used to join tables are missing, set to null character
  if (!"rank" %in% names(df)) df$rank <- NA_character_
  if (!"name" %in% names(df)) df$name <- NA_character_
  if (!"id"   %in% names(df)) df$id   <- NA_character_

  # Enforce string dtype
  df$id <- as.character(df$id)

  df
}

# Attempt to get node features (taxonomic data) for a species/node name
get_classification_fallback <- function(species_name, verbose = FALSE) {
    # Convert to binomial nomenclature if possible
    bn_name <- resolve_name(species_name, verbose)

    if (!nzchar(bn_name))
        return(
            tibble(
                name = NA, 
                rank = NA, 
                id = NA,
                species_name = species_name,
                source = NA
            )
        )

    # 1) Attempt lookup in GBIF
    if (verbose) cat("  • GBIF →", bn_name, "\n")
    gbif_id <- first_or_na(
        tryCatch(
            get_gbifid(bn_name, accepted = TRUE, rows = 1, messages = FALSE),
            error = function(e) NA
        )
    )
    if (!is.na(gbif_id)) {
        df <- tryCatch(
            classification(gbif_id, db = "gbif")[[1]],
            error = function(e) NULL
        ) |> clean_classification_df()
        if (nrow(df))
            return(
                mutate(df, species_name = species_name, source = "GBIF")
            )
    }

    # 2) Attempt lookup in ITIS if GBIF fails
    if (verbose) cat("  • ITIS →", bn_name, "\n")
    itis_id <- first_or_na(
        tryCatch(
            get_tsn(bn_name, accepted = TRUE, rows = 1, messages = FALSE),
            error = function(e) NA
        )
    )
    if (!is.na(itis_id)) {
        df <- tryCatch(
            classification(itis_id, db = "itis")[[1]],
            error = function(e) NULL
        ) |> clean_classification_df()
    if (!is.null(df) && nrow(df))
        return(
            mutate(df, species_name = species_name, source = "ITIS")
        )
  }

    # 3) Attempt lookup in NCBI if GBIF/ITIS fail
    if (verbose) cat("  • NCBI →", bn_name, "\n")
    ncbi_id <- first_or_na(
        tryCatch(
            get_uid(bn_name, rows = 1, messages = FALSE),
            error = function(e) NA
        )
    )
    if (!is.na(ncbi_id)) {
        df <- tryCatch(
            classification(ncbi_id, db = "ncbi")[[1]],
            error = function(e) NULL
        ) |> clean_classification_df()
    if (!is.null(df) && nrow(df))
        return(
            mutate(df, species_name = species_name, source = "NCBI")
        )
    }

    # 4) Attempt lookup in WoRMS if GBIF/ITIS/NCBI fail
    if (verbose) cat("  • WoRMS →", bn_name, "\n")
    worms_id <- first_or_na(
        tryCatch(
            get_wormsid(bn_name, rows = 1, messages = FALSE),
            error = function(e) NA
        )
    )
    if (!is.na(worms_id)) {
        df <- tryCatch(
            classification(worms_id, db = "worms")[[1]],
            error = function(e) NULL
        ) |> clean_classification_df()
    if (!is.null(df) && nrow(df))
        return(
            mutate(df, species_name = species_name, source = "WoRMS")
        )
    }

    # 5) If previous attempts fail, try EoL
    if (verbose) cat("  • EoL →", bn_name, "\n")
    eol_id <- first_or_na(
        tryCatch(
            get_eolid(bn_name, rows = 1, messages = FALSE),
            error = function(e) NA
        )
    )
  if (!is.na(eol_id)) {
    df <- tryCatch(
        classification(eol_id, db = "eol")[[1]],
        error = function(e) NULL
    ) |> clean_classification_df()
    if (!is.null(df) && nrow(df))
        return(
            mutate(df, species_name = species_name, source = "EoL")
        )
    }

    # If all attempts fail, return empty tibble
    tibble(
        name = NA,
        rank = NA,
        id = NA,
        species_name = species_name,
        source = NA
    )
}

# Create node features for a given CSV food web
process_species_csv <- function(csv_path, verbose = FALSE) {
    df <- read.csv(csv_path, stringsAsFactors = FALSE)
    species_col <- case_when(
        "Species" %in% names(df) ~ "Species",
        "Specie"  %in% names(df) ~ "Specie",
        TRUE ~ NA_character_
    )
    if (is.na(species_col)) 
        return(NULL)

    class_df <- bind_rows(
        lapply(
            unique(df[[species_col]]),
            get_classification_fallback,
            verbose = verbose
        )
    )

    wide <- class_df %>% 
        select(species_name, rank, name) %>% 
        distinct() %>%
        pivot_wider(
            names_from = rank,
            values_from = name,
            values_fn = ~paste(unique(.x), collapse = "; ")
        )

    nodes <- df %>% 
        rename(species_name = !!species_col) %>%
        left_join(wide, by = "species_name") %>% 
        relocate(species_name) %>%
        split(seq_len(nrow(.))) %>% 
        lapply(as.list)

    list(`edge matrix` = list(), `node features` = nodes)
}

# Create JSON node feature datasets for all CSV food webs in a folder
create_node_feature_sets <- function(
    data_dir = "csv_node_features",
    output_dir = "json_node_features",
    verbose = FALSE
) {
    csv_files <- list.files(data_dir, "\\.csv$", full.names = TRUE)
    if (!dir.exists(output_dir))
        dir.create(output_dir, TRUE)

    lapply(csv_files, function(fp) {
        res <- process_species_csv(fp, verbose)
        if (is.null(res)) 
            return(NULL)
        out <- file.path(
            output_dir,
            paste0(tools::file_path_sans_ext(basename(fp)), ".json"))
        write_json(res, out, pretty = TRUE, auto_unbox = TRUE)
        if (verbose) cat("✓", basename(fp), "\n")
    })
}

create_node_feature_sets(verbose = TRUE)