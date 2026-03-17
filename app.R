# ============================================================
# PK/PD/Safety Simulation Shiny App — bslib Modern UI
# DWJ1691 (TMDD) + Wegovy (Semaglutide) - Integrated Model
# Demo Version - Placeholder Parameters
# ============================================================

library(shiny)
library(bslib)
library(bsicons)
library(deSolve)
library(ggplot2)
library(plotly)
library(dplyr)
library(tidyr)
library(DT)

# ============================================================
# 1. MODEL PARAMETERS
# ============================================================

default_params <- list(
  sema_ka   = 0.02,
  sema_CL   = 0.066,
  sema_V1   = 3.5,
  sema_Q    = 0.12,
  sema_V2   = 7.0,
  sema_F    = 0.89,
  dwj_ka    = 0.008,
  dwj_F     = 0.75,
  dwj_CL    = 0.010,
  dwj_V1    = 3.0,
  dwj_Q     = 0.05,
  dwj_V2    = 6.0,
  dwj_kon   = 0.091,
  dwj_koff  = 0.001,
  dwj_kint  = 0.005,
  dwj_ksyn  = 1.0,
  dwj_kdeg  = 0.05,
  bw_base   = 100,
  bw_kin    = 0.0001,
  bw_kout   = 0.0001,
  bw_Emax_s = 0.8,
  bw_EC50_s = 50,
  bw_Emax_d = 0.6,
  bw_EC50_d = 20,
  gi_E0     = 0.05,
  gi_Emax   = 0.95,
  gi_EC50   = 80,
  gi_hill   = 1.5
)

# ============================================================
# 2. ODE SYSTEM
# ============================================================

pkpd_ode <- function(time, state, parms) {
  with(as.list(c(state, parms)), {
    dA_sema_depot <- -ka_s * A_sema_depot
    dA_sema_c     <- ka_s * F_s * A_sema_depot -
                     (CL_s/V1_s + Q_s/V1_s) * A_sema_c +
                     (Q_s/V2_s) * A_sema_p
    dA_sema_p     <- (Q_s/V1_s) * A_sema_c - (Q_s/V2_s) * A_sema_p
    C_sema_ugL    <- (A_sema_c / V1_s) * 1000

    C_dwj_free    <- A_dwj_c / V1_d
    dA_dwj_depot  <- -ka_d * A_dwj_depot
    dA_dwj_c      <- ka_d * F_d * A_dwj_depot -
                     (CL_d/V1_d) * A_dwj_c -
                     (Q_d/V1_d)  * A_dwj_c +
                     (Q_d/V2_d)  * A_dwj_p -
                     kon * C_dwj_free * R_free * V1_d +
                     koff * RC * V1_d
    dA_dwj_p      <- (Q_d/V1_d) * A_dwj_c - (Q_d/V2_d) * A_dwj_p
    dR_free       <- ksyn - kdeg * R_free - kon * C_dwj_free * R_free + koff * RC
    dRC           <- kon * C_dwj_free * R_free - koff * RC - kint * RC
    C_dwj_ugL     <- C_dwj_free * 1000

    IS   <- Emax_s * C_sema_ugL / (EC50_s + C_sema_ugL)
    ID   <- Emax_d * C_dwj_ugL  / (EC50_d + C_dwj_ugL)
    Icomb <- 1 - (1 - IS) * (1 - ID)
    dBW  <- kin_bw * (1 - Icomb) - kout_bw * BW

    C_pk  <- C_sema_ugL + 0.5 * C_dwj_ugL
    GI_rate <- E0_gi + (Emax_gi - E0_gi) * C_pk^hill_gi /
               (EC50_gi^hill_gi + C_pk^hill_gi)

    list(c(dA_sema_depot, dA_sema_c, dA_sema_p,
           dA_dwj_depot,  dA_dwj_c,  dA_dwj_p,
           dR_free, dRC, dBW),
         C_sema_ugL = C_sema_ugL,
         C_dwj_ugL  = C_dwj_ugL,
         GI_rate    = GI_rate)
  })
}

# ============================================================
# 3. SIMULATION HELPERS
# ============================================================

build_events <- function(schedule) {
  events <- lapply(schedule, function(ev) {
    data.frame(var=ev$state_var, time=ev$times_h,
               value=ev$dose_mg, method="add")
  })
  df <- do.call(rbind, events)
  df[order(df$time), ]
}

sema_weekly <- function(start_day, n_weeks, dose_mg) {
  list(drug="semaglutide",
       dose_mg = dose_mg * 1000,
       times_h = seq(start_day*24, (start_day+(n_weeks-1)*7)*24, by=7*24),
       state_var="A_sema_depot")
}

dwj_monthly <- function(start_day, n_months, dose_mg) {
  list(drug="DWJ1691",
       dose_mg = dose_mg * 1000,
       times_h = seq(start_day*24, (start_day+(n_months-1)*28)*24, by=28*24),
       state_var="A_dwj_depot")
}

make_cohort <- function(cohort_id, dwj_dose_mg=10) {
  switch(cohort_id,
    "Reference" = list(
      sema_weekly(0,4,0.25), sema_weekly(28,4,0.50),
      sema_weekly(56,4,1.00), sema_weekly(84,4,1.70),
      sema_weekly(112,4,2.40)
    ),
    "Cohort I (W-W-T-W-W)" = list(
      sema_weekly(0,4,0.25), sema_weekly(28,4,0.50),
      dwj_monthly(56,1,dwj_dose_mg),
      sema_weekly(84,4,1.70), sema_weekly(112,4,2.40)
    ),
    "Cohort II (W-W-W-T-W)" = list(
      sema_weekly(0,4,0.25), sema_weekly(28,4,0.50),
      sema_weekly(56,4,1.00), dwj_monthly(84,1,dwj_dose_mg),
      sema_weekly(112,4,2.40)
    ),
    "Cohort III (W-W-W-W-T)" = list(
      sema_weekly(0,4,0.25), sema_weekly(28,4,0.50),
      sema_weekly(56,4,1.00), sema_weekly(84,4,1.70),
      dwj_monthly(112,1,dwj_dose_mg)
    )
  )
}

run_simulation <- function(cohort_schedule, params, sim_days=175) {
  p <- params
  parms <- c(
    ka_s=p$sema_ka, CL_s=p$sema_CL, V1_s=p$sema_V1,
    Q_s=p$sema_Q,   V2_s=p$sema_V2, F_s=p$sema_F,
    ka_d=p$dwj_ka,  CL_d=p$dwj_CL,  V1_d=p$dwj_V1,
    Q_d=p$dwj_Q,    V2_d=p$dwj_V2,  F_d=p$dwj_F,
    kon=p$dwj_kon,  koff=p$dwj_koff, kint=p$dwj_kint,
    ksyn=p$dwj_ksyn, kdeg=p$dwj_kdeg,
    kin_bw=p$bw_kin, kout_bw=p$bw_kout,
    Emax_s=p$bw_Emax_s, EC50_s=p$bw_EC50_s,
    Emax_d=p$bw_Emax_d, EC50_d=p$bw_EC50_d,
    E0_gi=p$gi_E0, Emax_gi=p$gi_Emax,
    EC50_gi=p$gi_EC50, hill_gi=p$gi_hill
  )
  R0 <- p$dwj_ksyn / p$dwj_kdeg
  state0 <- c(A_sema_depot=0, A_sema_c=0, A_sema_p=0,
               A_dwj_depot=0,  A_dwj_c=0,  A_dwj_p=0,
               R_free=R0, RC=0, BW=p$bw_base)
  evts <- build_events(cohort_schedule)
  times <- seq(0, sim_days*24, by=1)
  out <- ode(y=state0, times=times, func=pkpd_ode, parms=parms,
             events=list(data=evts), method="lsoda")
  df <- as.data.frame(out)
  df$time_weeks <- df$time / (24*7)
  df
}

# ============================================================
# 4. THEME & COLORS
# ============================================================

app_theme <- bs_theme(
  version    = 5,
  bootswatch = "litera",
  primary    = "#2166ac",
  success    = "#1a9641",
  danger     = "#d73027",
  warning    = "#e6a817",
  base_font  = font_google("Inter"),
  heading_font = font_google("Inter"),
  "border-radius" = "0.5rem",
  "card-border-color" = "#e0e4ea"
) |>
  bs_add_rules("
    .sidebar { background: #f8fafc !important; border-right: 1px solid #e0e4ea; }
    .card { box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .card-header { font-weight: 600; font-size: 0.85rem;
                   letter-spacing: 0.03em; text-transform: uppercase;
                   background: #f8fafc; border-bottom: 1px solid #e0e4ea; }
    .value-box { border-radius: 0.5rem; }
    .nav-pills .nav-link.active { background-color: #2166ac; }
    .param-section { background: #f8fafc; border-radius: 0.5rem;
                     padding: 12px; margin-bottom: 8px; }
    .param-section h6 { font-size: 0.75rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;
                        color: #6c757d; margin-bottom: 10px; }
    .shiny-notification { border-radius: 0.5rem; }
    body { background: #f0f4f8; }
    .main-content { padding: 1.2rem; }
  ")

cohort_colors <- c(
  "Reference"               = "#888888",
  "Cohort I (W-W-T-W-W)"   = "#2166ac",
  "Cohort II (W-W-W-T-W)"  = "#1a9641",
  "Cohort III (W-W-W-W-T)" = "#d73027"
)

# ============================================================
# 5. UI
# ============================================================

ui <- page_sidebar(
  theme = app_theme,
  title = tags$span(
    tags$img(src="https://www.r-project.org/logo/Rlogo.svg",
             height="22px", style="margin-right:8px; vertical-align:middle;"),
    "DWJ1691 + Wegovy  |  PK/PD Simulator",
    style = "font-size:1rem; font-weight:600; color:#1e293b;"
  ),

  # ---- SIDEBAR ----
  sidebar = sidebar(
    width = 280,
    bg = "#f8fafc",

    # Cohort selector
    div(class="param-section",
      tags$h6("Cohort Selection"),
      checkboxGroupInput("cohorts", NULL,
        choices  = names(cohort_colors),
        selected = names(cohort_colors)
      )
    ),

    # DWJ1691 dose
    div(class="param-section",
      tags$h6("DWJ1691 Dose"),
      sliderInput("dwj_dose", "Monthly SC dose (mg)",
                  min=1, max=50, value=10, step=1, ticks=FALSE)
    ),

    # Simulation length
    div(class="param-section",
      tags$h6("Simulation"),
      sliderInput("sim_weeks", "Duration (weeks)",
                  min=12, max=36, value=25, step=1, ticks=FALSE)
    ),

    hr(style="margin:8px 0; border-color:#e0e4ea;"),

    actionButton("run_sim", "Run Simulation",
                 icon = icon("play-circle"),
                 class = "btn-primary w-100 fw-semibold"),

    br(),

    # Download
    downloadButton("dl_csv", "Download CSV",
                   class = "btn-outline-secondary w-100 btn-sm")
  ),

  # ---- MAIN CONTENT ----
  div(class="main-content",

    # KPI row
    layout_columns(
      col_widths = c(3,3,3,3),
      value_box(
        title = "Semaglutide C\u2098\u2090\u2093",
        value = textOutput("kpi_scmax", inline=TRUE),
        showcase = bs_icon("arrow-up-circle-fill"),
        theme = "primary", height = "100px"
      ),
      value_box(
        title = "DWJ1691 C\u2098\u2090\u2093",
        value = textOutput("kpi_dcmax", inline=TRUE),
        showcase = bs_icon("arrow-up-circle-fill"),
        theme = "danger", height = "100px"
      ),
      value_box(
        title = "Max BW loss",
        value = textOutput("kpi_bw", inline=TRUE),
        showcase = bs_icon("activity"),
        theme = "success", height = "100px"
      ),
      value_box(
        title = "Peak GI AE rate",
        value = textOutput("kpi_gi", inline=TRUE),
        showcase = bs_icon("exclamation-triangle-fill"),
        theme = "warning", height = "100px"
      )
    ),

    br(),

    # Tab panels for charts
    navset_card_pill(
      id = "main_tabs",

      nav_panel("PK Profile",
        icon = bs_icon("graph-up"),
        card_body(
          p(class="text-muted", style="font-size:0.8rem; margin-bottom:4px;",
            "Plasma concentration over time — solid: Semaglutide, dashed: DWJ1691"),
          plotlyOutput("pk_plot", height="380px")
        )
      ),

      nav_panel("Body Weight",
        icon = bs_icon("person"),
        card_body(
          plotlyOutput("bw_plot", height="380px")
        )
      ),

      nav_panel("GI Safety",
        icon = bs_icon("shield-exclamation"),
        card_body(
          plotlyOutput("gi_plot", height="380px")
        )
      ),

      nav_panel("Combined View",
        icon = bs_icon("grid"),
        card_body(
          layout_columns(
            col_widths = c(12),
            plotlyOutput("pk_plot2",  height="240px"),
          ),
          layout_columns(
            col_widths = c(6,6),
            plotlyOutput("bw_plot2",  height="220px"),
            plotlyOutput("gi_plot2",  height="220px")
          )
        )
      ),

      nav_panel("PK Summary Table",
        icon = bs_icon("table"),
        card_body(
          DTOutput("pk_table")
        )
      ),

      nav_panel("Parameters",
        icon = bs_icon("sliders"),
        card_body(
          layout_columns(
            col_widths = c(6,6),

            # Wegovy PK
            card(
              card_header("Wegovy (Semaglutide) PK"),
              card_body(
                numericInput("sema_CL","CL (L/h)",   value=0.066, step=0.001, width="100%"),
                numericInput("sema_V1","V1 (L)",     value=3.5,   step=0.1,   width="100%"),
                numericInput("sema_ka","ka (h⁻¹)",  value=0.02,  step=0.001, width="100%"),
                numericInput("sema_F", "F (0–1)",    value=0.89,  step=0.01,  width="100%")
              )
            ),

            # DWJ1691 PK
            card(
              card_header("DWJ1691 PK — TMDD"),
              card_body(
                numericInput("dwj_CL",  "CL (L/h)",          value=0.010, step=0.001, width="100%"),
                numericInput("dwj_V1",  "V1 (L)",            value=3.0,   step=0.1,   width="100%"),
                numericInput("dwj_kon", "kon (L/nmol/h)",    value=0.091, step=0.001, width="100%"),
                numericInput("dwj_koff","koff (h⁻¹)",       value=0.001, step=0.0001,width="100%"),
                numericInput("dwj_kint","kint (h⁻¹)",       value=0.005, step=0.001, width="100%"),
                numericInput("dwj_ksyn","R synthesis (nmol/L/h)", value=1.0, step=0.1, width="100%")
              )
            ),

            # PD
            card(
              card_header("Body Weight PD"),
              card_body(
                numericInput("bw_Emax_s","Emax Sema",          value=0.8, step=0.05, width="100%"),
                numericInput("bw_EC50_s","EC50 Sema (µg/L)",   value=50,  step=5,    width="100%"),
                numericInput("bw_Emax_d","Emax DWJ1691",       value=0.6, step=0.05, width="100%"),
                numericInput("bw_EC50_d","EC50 DWJ1691 (µg/L)",value=20,  step=5,    width="100%")
              )
            ),

            # Safety
            card(
              card_header("GI AE Safety Model"),
              card_body(
                numericInput("gi_E0",   "Baseline GI rate", value=0.05, step=0.01, width="100%"),
                numericInput("gi_Emax", "Max GI rate",      value=0.95, step=0.05, width="100%"),
                numericInput("gi_EC50", "EC50 (µg/L)",      value=80,   step=5,    width="100%"),
                numericInput("gi_hill", "Hill coeff.",      value=1.5,  step=0.1,  width="100%")
              )
            )
          )
        )
      )
    )
  )
)

# ============================================================
# 6. SERVER
# ============================================================

server <- function(input, output, session) {

  get_params <- reactive({
    p <- default_params
    p$sema_CL    <- input$sema_CL
    p$sema_V1    <- input$sema_V1
    p$sema_ka    <- input$sema_ka
    p$sema_F     <- input$sema_F
    p$dwj_CL     <- input$dwj_CL
    p$dwj_V1     <- input$dwj_V1
    p$dwj_kon    <- input$dwj_kon
    p$dwj_koff   <- input$dwj_koff
    p$dwj_kint   <- input$dwj_kint
    p$dwj_ksyn   <- input$dwj_ksyn
    p$bw_Emax_s  <- input$bw_Emax_s
    p$bw_EC50_s  <- input$bw_EC50_s
    p$bw_Emax_d  <- input$bw_Emax_d
    p$bw_EC50_d  <- input$bw_EC50_d
    p$gi_E0      <- input$gi_E0
    p$gi_Emax    <- input$gi_Emax
    p$gi_EC50    <- input$gi_EC50
    p$gi_hill    <- input$gi_hill
    p
  })

  sim_data <- eventReactive(input$run_sim, {
    req(input$cohorts)
    withProgress(message="Running ODE simulation…", value=0, {
      results <- lapply(input$cohorts, function(coh) {
        incProgress(1/length(input$cohorts), detail=coh)
        sched <- make_cohort(coh, dwj_dose_mg=input$dwj_dose)
        df    <- run_simulation(sched, get_params(),
                                sim_days=input$sim_weeks*7)
        df$cohort <- coh
        df
      })
    })
    bind_rows(results)
  }, ignoreNULL=FALSE)

  # Auto-run on startup
  observe({ if(is.null(sim_data())) shinyjs::click("run_sim") })

  # ---- Shared plot theme ----
  pk_theme <- function() {
    theme_minimal(base_size=12, base_family="sans") +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color="#e8ecf0", linewidth=0.4),
      axis.title       = element_text(size=11, color="#4a5568"),
      legend.position  = "top",
      legend.text      = element_text(size=10),
      plot.background  = element_rect(fill="white", color=NA)
    )
  }

  make_pk_plot <- function(df) {
    df_long <- df |>
      select(cohort, time_weeks, C_sema_ugL, C_dwj_ugL) |>
      pivot_longer(c(C_sema_ugL, C_dwj_ugL), names_to="drug", values_to="conc") |>
      mutate(drug = ifelse(drug=="C_sema_ugL","Semaglutide","DWJ1691"))
    p <- ggplot(df_long, aes(x=time_weeks, y=conc, color=cohort, linetype=drug)) +
      geom_line(linewidth=0.9, alpha=0.85) +
      scale_color_manual(values=cohort_colors, name="Cohort") +
      scale_linetype_manual(values=c("Semaglutide"="solid","DWJ1691"="dashed"), name="Drug") +
      labs(x="Time (week)", y="Plasma concentration (µg/L)") +
      pk_theme()
    ggplotly(p) |> layout(hovermode="x unified",
                           legend=list(orientation="h", y=1.12))
  }

  make_bw_plot <- function(df, bw0=100) {
    p <- ggplot(df, aes(x=time_weeks, y=(BW-bw0)/bw0*100, color=cohort)) +
      geom_line(linewidth=1, alpha=0.85) +
      scale_color_manual(values=cohort_colors, name="Cohort") +
      labs(x="Time (week)", y="Body weight change (%BW)") +
      pk_theme()
    ggplotly(p) |> layout(hovermode="x unified",
                           legend=list(orientation="h", y=1.12))
  }

  make_gi_plot <- function(df) {
    p <- ggplot(df, aes(x=time_weeks, y=GI_rate*100, color=cohort)) +
      geom_line(linewidth=1, alpha=0.85) +
      scale_color_manual(values=cohort_colors, name="Cohort") +
      labs(x="Time (week)", y="GI AE rate (%)") +
      ylim(0,100) +
      pk_theme()
    ggplotly(p) |> layout(hovermode="x unified",
                           legend=list(orientation="h", y=1.12))
  }

  output$pk_plot  <- renderPlotly({ req(sim_data()); make_pk_plot(sim_data()) })
  output$bw_plot  <- renderPlotly({ req(sim_data()); make_bw_plot(sim_data()) })
  output$gi_plot  <- renderPlotly({ req(sim_data()); make_gi_plot(sim_data()) })
  output$pk_plot2 <- renderPlotly({ req(sim_data()); make_pk_plot(sim_data()) })
  output$bw_plot2 <- renderPlotly({ req(sim_data()); make_bw_plot(sim_data()) })
  output$gi_plot2 <- renderPlotly({ req(sim_data()); make_gi_plot(sim_data()) })

  # ---- KPI outputs ----
  output$kpi_scmax <- renderText({
    req(sim_data())
    paste0(round(max(sim_data()$C_sema_ugL, na.rm=TRUE), 1), " µg/L")
  })
  output$kpi_dcmax <- renderText({
    req(sim_data())
    paste0(round(max(sim_data()$C_dwj_ugL, na.rm=TRUE), 1), " µg/L")
  })
  output$kpi_bw <- renderText({
    req(sim_data())
    bw0 <- default_params$bw_base
    paste0(round(min((sim_data()$BW - bw0)/bw0*100, na.rm=TRUE), 1), "%")
  })
  output$kpi_gi <- renderText({
    req(sim_data())
    paste0(round(max(sim_data()$GI_rate*100, na.rm=TRUE), 1), "%")
  })

  # ---- PK Summary Table ----
  output$pk_table <- renderDT({
    req(sim_data())
    bw0 <- default_params$bw_base
    tbl <- sim_data() |>
      group_by(Cohort = cohort) |>
      summarise(
        `Sema Cmax (µg/L)`  = round(max(C_sema_ugL, na.rm=TRUE), 2),
        `Sema Tmax (wk)`    = round(time_weeks[which.max(C_sema_ugL)], 1),
        `DWJ Cmax (µg/L)`   = round(max(C_dwj_ugL, na.rm=TRUE), 2),
        `DWJ Tmax (wk)`     = round(time_weeks[which.max(C_dwj_ugL)], 1),
        `Max BW loss (%)`   = round(min((BW-bw0)/bw0*100, na.rm=TRUE), 2),
        `Peak GI AE (%)`    = round(max(GI_rate*100, na.rm=TRUE), 1),
        .groups = "drop"
      )
    datatable(tbl,
              options = list(dom="t", pageLength=10, scrollX=TRUE),
              rownames = FALSE,
              class    = "table table-striped table-hover table-sm") |>
      formatStyle("Cohort", fontWeight="600") |>
      formatStyle("Max BW loss (%)",
                  color = styleInterval(c(-10,-5), c("#d73027","#e6a817","#1a9641")))
  })

  # ---- CSV Download ----
  output$dl_csv <- downloadHandler(
    filename = function() paste0("pkpd_sim_", Sys.Date(), ".csv"),
    content  = function(file) {
      write.csv(sim_data(), file, row.names=FALSE)
    }
  )
}

# ============================================================
# 7. RUN
# ============================================================

shinyApp(ui, server)
